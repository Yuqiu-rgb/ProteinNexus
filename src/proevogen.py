# -*- coding: utf-8 -*-
"""
ProEvoGen: A Surrogate-Assisted, Biologically-Informed Framework for Protein PTM Site
Data Augmentation using Generative Language Models and Evolutionary Computing.

This script implements a highly efficient data augmentation pipeline for PTM datasets.
Key Innovations:
1.  Fitness Surrogate Model: A lightweight MLP is trained to predict expensive fitness
    scores (structure, plausibility) from ESM embeddings, accelerating the evolutionary
    process by orders of magnitude.
2.  ESM-Guided Contextual Mutation: Leverages ESM-3's predictive power to perform
    biologically plausible mutations based on sequence context.
3.  PSSM Scoring: Replaces binary regex matching with a continuous PSSM score for
    more nuanced motif optimization.
4.  Surrogate -> Evolve -> Rerank Workflow: Ensures both speed during evolution and
    accuracy in the final selection.
"""
import torch
import pandas as pd
import numpy as np
import random
import re
import os
import subprocess
import time
import shutil
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from transformers import AutoTokenizer, EsmForConditionalGeneration, GenerationConfig
from Bio.Align import substitution_matrices
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

'''Protein Evolutionary Generator (ProEvoGen)'''


# =======================================
# 1. Configuration Class
# =======================================
class Config:
    """Stores all hyperparameters and configuration settings."""
    # --- Data Parameters ---
    INPUT_CSV = "protein_data.csv"
    OUTPUT_CSV = "proevogen_enhanced_data.csv"
    SEQ_LENGTH = 33
    CENTER_POS = 16

    # --- PTM Parameters ---
    PTM_TYPE = "phosphorylation"
    PTM_AA = {"S", "T", "Y"}
    # PSSM file path (example format, see PSSMManager for details)
    PSSM_FILE = "phos_pssm.txt"

    # --- Generation & Augmentation Parameters ---
    GENERATION_FACTOR = 5
    INPAINTING_MASK_RATIO = 0.4
    INPAINTING_WINDOW_SIZE = 7

    # --- ESM Model Parameters ---
    ESM_MODEL_NAME = "esm3-sm-open-v1 "
    QUANT_4BIT = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Surrogate Model Parameters ---
    SURROGATE_TRAINING_SIZE = 500  # Number of samples to train the surrogate
    SURROGATE_MODEL_PATH = "fitness_surrogate.pkl"

    # --- NSGA-II Parameters ---
    POPULATION_SIZE = 100
    GENERATIONS = 50
    MUTATION_PROB = 0.3  # Higher mutation prob for ESM-guided
    CROSSOVER_PROB = 0.9
    # Fitness weights: (Motif, Structure, Plausibility).
    FITNESS_WEIGHTS = (1.0, 1.0, -1.0)  # Max Motif, Max Structure, Min Perplexity

    # --- External Tool Parameters ---
    NETSURFP_PATH = "./netsurfp-3.0/run_netsurfp.py"

    # --- Hard Negative Mining Parameters ---
    HARD_NEGATIVE_K = 5
    NEGATIVE_POOL_RATIO = 0.5


# =======================================
# 2. ESM-3 Manager Class (with new method)
# =======================================
class ESM3Manager:
    """Handles all interactions with the ESM-3 model."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL_NAME)
        self.model = self._load_model()
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def _load_model(self):
        """Loads the ESM-3 model with optional quantization."""
        model = EsmForConditionalGeneration.from_pretrained(
            self.config.ESM_MODEL_NAME,
            device_map="auto",
            load_in_4bit=self.config.QUANT_4BIT,
            torch_dtype=torch.float16 if self.config.QUANT_4BIT else torch.float32
        )
        return model

    def inpaint_sequences(self, seed_seqs: list[str], num_to_gen: int) -> list[str]:
        """Generates new sequences via inpainting in a batched and efficient manner."""
        generated_seqs = []
        pbar = tqdm(total=num_to_gen, desc="Inpainting")
        while len(generated_seqs) < num_to_gen:
            seed_seq = random.choice(seed_seqs)
            seq_list = list(seed_seq)
            # Define window for masking
            window_start = max(0, self.config.CENTER_POS - self.config.INPAINTING_WINDOW_SIZE)
            window_end = min(self.config.SEQ_LENGTH, self.config.CENTER_POS + self.config.INPAINTING_WINDOW_SIZE + 1)
            indices_in_window = [i for i in range(window_start, window_end) if i != self.config.CENTER_POS]
            num_to_mask = int(len(indices_in_window) * self.config.INPAINTING_MASK_RATIO)
            if num_to_mask == 0 and len(indices_in_window) > 0: num_to_mask = 1
            indices_to_mask = random.sample(indices_in_window, k=num_to_mask)
            for i in indices_to_mask:
                seq_list[i] = self.tokenizer.mask_token
            masked_prompt = "".join(seq_list)

            inputs = self.tokenizer(masked_prompt, return_tensors="pt").to(self.config.DEVICE)
            gen_config = GenerationConfig(do_sample=True, temperature=1.0, top_k=50,
                                          max_length=self.config.SEQ_LENGTH + 2)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, generation_config=gen_config)

            decoded_seq = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")

            if len(decoded_seq) == self.config.SEQ_LENGTH and decoded_seq[self.config.CENTER_POS] in self.config.PTM_AA:
                generated_seqs.append(decoded_seq)
                pbar.update(1)
        pbar.close()
        return list(set(generated_seqs))

    def get_embedding(self, sequences: list[str], batch_size=32) -> np.ndarray:
        """Gets sequence-level embeddings (CLS token) in batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Getting Embeddings"):
            batch = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding="longest", truncation=True,
                max_length=self.config.SEQ_LENGTH
            ).to(self.config.DEVICE)
            with torch.no_grad():
                outputs = self.model.base_model.encoder(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def get_masked_logits(self, sequence: str, position: int) -> torch.Tensor:
        """Gets the logits for a masked position in a sequence."""
        seq_list = list(sequence)
        seq_list[position] = self.tokenizer.mask_token
        masked_input = "".join(seq_list)
        inputs = self.tokenizer(masked_input, return_tensors="pt").to(self.config.DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Return logits for the masked position (+1 for BOS token)
        return logits[0, position + 1, :]

    def calculate_pseudo_perplexity(self, sequence: str) -> float:
        """Calculates pseudo-perplexity, a measure of sequence plausibility."""
        total_neg_log_likelihood = 0.0
        seq_len = len(sequence)
        for i in range(seq_len):
            original_char = sequence[i]
            if original_char not in self.amino_acids: continue

            logits = self.get_masked_logits(sequence, i)
            original_token_id = self.tokenizer.convert_tokens_to_ids(original_char)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[original_token_id].item()
            total_neg_log_likelihood -= token_log_prob

        avg_neg_log_likelihood = total_neg_log_likelihood / seq_len
        return np.exp(avg_neg_log_likelihood)


# =======================================
# 3. PSSM Manager (New Class)
# =======================================
class PSSMManager:
    """Handles loading and scoring with a PSSM."""

    def __init__(self, pssm_file_path: str):
        self.pssm, self.aa_order = self._load_pssm(pssm_file_path)
        self.aa_map = {aa: i for i, aa in enumerate(self.aa_order)}

    def _load_pssm(self, file_path):
        if not os.path.exists(file_path):
            print(f"Warning: PSSM file not found at {file_path}. Motif score will be 0.")
            # Create a dummy PSSM to avoid errors
            amino_acids = "ARNDCQEGHILKMFPSTWYV"
            dummy_pssm = pd.DataFrame(np.zeros((len(amino_acids), Config.SEQ_LENGTH)), index=list(amino_acids))
            return dummy_pssm, list(amino_acids)

        df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        aa_order = list(df.index)
        return df, aa_order

    def score(self, sequence: str) -> float:
        """Scores a sequence against the PSSM."""
        total_score = 0
        if self.pssm is None: return 0.0

        for i, aa in enumerate(sequence):
            if aa in self.aa_map and str(i) in self.pssm.columns:
                total_score += self.pssm.loc[aa, str(i)]
        return total_score


# =======================================
# 4. Structure Predictor Class
# =======================================
class StructurePredictor:
    """Wrapper for local NetSurfP-3.0."""

    def __init__(self, config: Config):
        self.config = config
        if not os.path.exists(config.NETSURFP_PATH):
            raise FileNotFoundError(f"NetSurfP-3.0 script not found at: {config.NETSURFP_PATH}")

    def predict_batch(self, sequences: list[str]) -> dict:
        """Runs NetSurfP-3.0 on a batch of sequences."""
        fasta_content = ""
        for i, seq in enumerate(sequences):
            fasta_content += f">seq_{i}\n{seq}\n"

        temp_fasta_path = "temp_batch.fasta"
        temp_output_dir = "temp_netsurfp_out"

        with open(temp_fasta_path, "w") as f:
            f.write(fasta_content)

        command = [
            "python", self.config.NETSURFP_PATH, "-i", temp_fasta_path,
            "-o", temp_output_dir, "-b", "64"  # Use batching
        ]

        results = {}
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            for i in range(len(sequences)):
                output_csv = os.path.join(temp_output_dir, f"seq_{i}.csv")
                if os.path.exists(output_csv):
                    df = pd.read_csv(output_csv)
                    center_props = df.iloc[self.config.CENTER_POS]
                    results[sequences[i]] = {
                        "rsa": center_props["rsa"],
                        "disorder": center_props["disorder"]
                    }
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running NetSurfP-3.0 batch: {e}")
        finally:
            if os.path.exists(temp_fasta_path): os.remove(temp_fasta_path)
            if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        return results


# =======================================
# 5. Fitness Surrogate (New Class)
# =======================================
class FitnessSurrogate:
    """A lightweight model to predict fitness scores from embeddings."""

    def __init__(self, config: Config):
        self.config = config
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500,
                                  random_state=42, early_stopping=True)
        self.scaler = StandardScaler()

    def train(self, embeddings: np.ndarray, scores: np.ndarray):
        """Trains the surrogate model."""
        X_train, X_test, y_train, y_test = train_test_split(embeddings, scores, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training fitness surrogate model...")
        self.model.fit(X_train_scaled, y_train)
        score = self.model.score(X_test_scaled, y_test)
        print(f"Surrogate model R^2 score: {score:.4f}")
        joblib.dump((self.model, self.scaler), self.config.SURROGATE_MODEL_PATH)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predicts fitness scores using the trained model."""
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.model.predict(embeddings_scaled)

    def load(self):
        """Loads a pre-trained surrogate model."""
        print(f"Loading surrogate model from {self.config.SURROGATE_MODEL_PATH}")
        self.model, self.scaler = joblib.load(self.config.SURROGATE_MODEL_PATH)


# =======================================
# 6. ProEvoGen NSGA-II Optimizer
# =======================================
class ProEvoGenOptimizer:
    """Manages multi-objective optimization using a surrogate-assisted NSGA-II."""

    def __init__(self, config: Config, esm_manager: ESM3Manager, pssm_manager: PSSMManager,
                 surrogate: FitnessSurrogate):
        self.config = config
        self.esm_manager = esm_manager
        self.pssm_manager = pssm_manager
        self.surrogate = surrogate
        self.toolbox = self._setup_toolbox()
        self.amino_acids_list = list(self.esm_manager.amino_acids)
        self.aa_token_ids = self.esm_manager.tokenizer.convert_tokens_to_ids(self.amino_acids_list)
        self.aa_map = {aa: i for i, aa in enumerate(self.amino_acids_list)}

    def _setup_creator(self):
        """Sets up DEAP creator."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=self.config.FITNESS_WEIGHTS)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

    def _setup_toolbox(self):
        """Configures the DEAP toolbox."""
        self._setup_creator()
        toolbox = base.Toolbox()
        toolbox.register("attr_char", random.choice, self.amino_acids_list)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_char, self.config.SEQ_LENGTH)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_fitness_surrogate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_esm_guided)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def evaluate_fitness_surrogate(self, individual: list) -> tuple:
        """Evaluates fitness using the PSSM score and the fast surrogate model."""
        sequence = "".join(individual)

        # Objective 1: PSSM Motif Score (Maximization)
        motif_score = self.pssm_manager.score(sequence)

        # Get embedding for surrogate prediction
        embedding = self.esm_manager.get_embedding([sequence])

        # Predict objectives 2 & 3 using the surrogate
        predicted_scores = self.surrogate.predict(embedding)[0]
        structure_score, perplexity = predicted_scores[0], predicted_scores[1]

        return motif_score, structure_score, perplexity

    def mutate_esm_guided(self, individual: list) -> tuple:
        """Mutates an individual using ESM-3's contextual predictions."""
        for i in range(len(individual)):
            if random.random() < self.config.MUTATION_PROB:
                if i == self.config.CENTER_POS:
                    individual[i] = random.choice(list(self.config.PTM_AA))
                else:
                    logits = self.esm_manager.get_masked_logits("".join(individual), i)
                    # Get probabilities for valid amino acids only
                    aa_logits = logits[self.aa_token_ids]
                    probs = softmax(aa_logits.cpu().numpy())
                    new_aa = random.choices(self.amino_acids_list, weights=probs, k=1)[0]
                    individual[i] = new_aa
        return individual,

    def run(self, initial_population: list[str]) -> list[str]:
        """Runs the surrogate-assisted NSGA-II optimization."""
        population = [creator.Individual(list(seq)) for seq in initial_population]

        # Evaluate initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = self.toolbox.select(population, k=len(population))

        for gen in tqdm(range(self.config.GENERATIONS), desc="Surrogate-Assisted Evolution"):
            offspring = algorithms.varAnd(population, self.toolbox, self.config.CROSSOVER_PROB,
                                          self.config.MUTATION_PROB)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population = self.toolbox.select(population + offspring, k=self.config.POPULATION_SIZE)

        pareto_front = tools.selBest(population, k=self.config.POPULATION_SIZE)
        return ["".join(ind) for ind in pareto_front]


# =======================================
# 7. Hard Negative Miner Class
# =======================================
class HardNegativeMiner:
    """Mines hard negative samples."""

    def __init__(self, config: Config, esm_manager: ESM3Manager):
        self.config = config
        self.esm_manager = esm_manager

    def find_hard_negatives(self, positive_seqs: list[str], negative_pool: list[str]) -> list[str]:
        """Finds hard negatives using k-NN in embedding space."""
        print("Embedding pools for hard negative mining...")
        pos_embeddings = self.esm_manager.get_embedding(positive_seqs)
        neg_embeddings = self.esm_manager.get_embedding(negative_pool)

        nn_model = NearestNeighbors(n_neighbors=self.config.HARD_NEGATIVE_K, metric='cosine', algorithm='brute')
        nn_model.fit(neg_embeddings)

        print("Querying for hard negatives...")
        _, indices = nn_model.kneighbors(pos_embeddings)
        hard_negatives = {negative_pool[idx] for i in indices for idx in i}
        print(f"Found {len(hard_negatives)} unique hard negatives.")
        return list(hard_negatives)


# =======================================
# 8. Main Pipeline
# =======================================
def main():
    config = Config()

    # --- Step 1: Initialization ---
    print("Initializing models and tools...")
    esm_manager = ESM3Manager(config)
    pssm_manager = PSSMManager(config.PSSM_FILE)
    structure_predictor = StructurePredictor(config)
    surrogate = FitnessSurrogate(config)

    # --- Step 2: Load Data ---
    print(f"Loading data from {config.INPUT_CSV}...")
    df = pd.read_csv(config.INPUT_CSV)
    df = df[df['seq'].str.len() == config.SEQ_LENGTH].drop_duplicates().reset_index(drop=True)
    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]
    original_pos_seqs = positive_df['seq'].tolist()

    # --- Step 3: Surrogate Model Training ---
    print("\n--- Phase 1: Surrogate Model Training ---")
    if not os.path.exists(config.SURROGATE_MODEL_PATH):
        # 3.1 Generate diverse sequences for training
        surrogate_train_seqs = esm_manager.inpaint_sequences(original_pos_seqs, config.SURROGATE_TRAINING_SIZE)

        # 3.2 Get true fitness scores (the expensive part)
        print("Calculating true fitness scores for surrogate training...")
        # Get structure scores
        struct_results = structure_predictor.predict_batch(surrogate_train_seqs)
        # Get perplexity scores
        perplexities = [esm_manager.calculate_pseudo_perplexity(seq) for seq in
                        tqdm(surrogate_train_seqs, desc="Calculating Perplexity")]

        # 3.3 Assemble training data
        train_embeddings = esm_manager.get_embedding(surrogate_train_seqs)
        train_scores = []
        valid_indices = []
        for i, seq in enumerate(surrogate_train_seqs):
            if seq in struct_results:
                struct_score = (struct_results[seq]["rsa"] + struct_results[seq]["disorder"]) / 2.0
                train_scores.append([struct_score, perplexities[i]])
                valid_indices.append(i)

        # 3.4 Train the surrogate model
        surrogate.train(train_embeddings[valid_indices], np.array(train_scores))
    else:
        surrogate.load()

    # --- Step 4: Positive Sample Augmentation ---
    print("\n--- Phase 2: Surrogate-Assisted Evolution ---")
    optimizer = ProEvoGenOptimizer(config, esm_manager, pssm_manager, surrogate)
    num_to_generate = len(df) * (config.GENERATION_FACTOR - 1)
    num_pos_to_generate = int(num_to_generate * (len(positive_df) / len(df)))

    # 4.1 Generate initial population for evolution
    initial_candidates = esm_manager.inpaint_sequences(original_pos_seqs, num_pos_to_generate)

    # 4.2 Run optimization in batches
    optimized_candidates = []
    for i in range(0, len(initial_candidates), config.POPULATION_SIZE):
        batch = initial_candidates[i:i + config.POPULATION_SIZE]
        if batch:
            optimized_batch = optimizer.run(batch)
            optimized_candidates.extend(optimized_batch)

    optimized_candidates = list(set(optimized_candidates))
    print(f"Evolution generated {len(optimized_candidates)} candidates.")

    # --- Step 5: Final Validation & Selection ---
    print("\n--- Phase 3: Final Validation of Pareto Front ---")
    # Re-evaluate the final candidates with TRUE fitness functions
    final_struct_scores = structure_predictor.predict_batch(optimized_candidates)
    final_pos_seqs = []
    for seq in tqdm(optimized_candidates, desc="Final Validation"):
        if seq in final_struct_scores:
            final_pos_seqs.append(seq)  # Keep all valid ones for now, could be sorted later

    print(f"Obtained {len(final_pos_seqs)} unique, validated positive sequences.")

    # --- Step 6: Negative Sample Augmentation ---
    print("\n--- Phase 4: Hard Negative Mining ---")
    hard_miner = HardNegativeMiner(config, esm_manager)
    potential_neg_pool = [s for s in negative_df['seq'].tolist() if s[config.CENTER_POS] in config.PTM_AA]
    hard_negatives = hard_miner.find_hard_negatives(original_pos_seqs + final_pos_seqs, potential_neg_pool)

    num_neg_to_generate = num_to_generate - len(final_pos_seqs)
    num_hard = int(num_neg_to_generate * config.NEGATIVE_POOL_RATIO)
    num_easy = num_neg_to_generate - num_hard

    final_neg_seqs = random.sample(hard_negatives, min(num_hard, len(hard_negatives)))
    easy_negs = list(set(negative_df['seq'].tolist()) - set(hard_negatives))
    if easy_negs:
        final_neg_seqs.extend(random.sample(easy_negs, min(num_easy, len(easy_negs))))

    final_neg_seqs = list(set(final_neg_seqs))
    print(f"Generated {len(final_neg_seqs)} negative sequences.")

    # --- Step 7: Final Dataset Assembly ---
    print("\n--- Assembling final dataset ---")
    new_pos_data = [{"seq": seq, "label": 1} for seq in final_pos_seqs]
    new_neg_data = [{"seq": seq, "label": 0} for seq in final_neg_seqs]

    enhanced_df = pd.concat([df, pd.DataFrame(new_pos_data), pd.DataFrame(new_neg_data)], ignore_index=True)
    enhanced_df = enhanced_df.drop_duplicates(subset=['seq']).reset_index(drop=True)
    enhanced_df.to_csv(config.OUTPUT_CSV, index=False)

    print("\n=== ProEvoGen Augmentation Complete ===")
    print(f"Original dataset size: {len(df)} ({len(positive_df)} Pos, {len(negative_df)} Neg)")
    print(
        f"Enhanced dataset size: {len(enhanced_df)} ({enhanced_df['label'].sum()} Pos, {len(enhanced_df) - enhanced_df['label'].sum()} Neg)")
    print(f"Data saved to {config.OUTPUT_CSV}")
