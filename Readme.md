# 			**ProteinNexus: An SE(3)-Equivariant and Dynamic Multi-Modal Engine Advancing from PTM Site Prediction to Network Understanding**

**1.Catalog description**

​	data: Data storage

​	src: Main code storage

​	model/embedding: Weight storage

------

**2.Requirements**

```
pandas

scikit-learn

transformers

matplotlib

umap-learn
```

Please download torch version>=2.0 or above to avoid version conflicts

------

**3.PLM model**

Please browse the required PLM official GitHub document to obtain the model embedding or weight.
ESM-3 and ESM-C : [evolutionaryscale/esm](https://github.com/evolutionaryscale/esm)

Saprot: [westlake-repl/SaProt: Saprot: Protein Language Model with Structural Alphabet (AA+3Di)](https://github.com/westlake-repl/SaProt)

The ESM series officially provides detailed usage tutorials.

Saport also offers comprehensive official tutorials. For one-click generation, you can visit SaprotHub:[westlake-repl/SaprotHub: Making Protein Language Modeling Accessible to All Biologists](https://github.com/westlake-repl/SaprotHub)


------

**4.Onedrive/Huggingface  description**

Due to GitHub's file size limit, we have uploaded the model weights and embeddings to OneDrive and the Hugging Face repository.

------

**5.How to Run****

Just run src/train.py to start training

```python
python train.py
```


