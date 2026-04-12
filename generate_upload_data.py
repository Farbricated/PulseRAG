"""
PulseRAG — Synthetic Upload Data Generator
Run: python generate_upload_data.py
Generates 3 files ready to upload in the Upload Docs tab:
  - ml_concepts.txt
  - deep_learning_notes.csv
  - rag_systems_guide.txt
"""

import csv, os

# ── File 1: ML Concepts TXT ───────────────────────────────────────────────────
ml_txt = """
Support Vector Machines (SVM) find the optimal hyperplane that maximizes the margin between classes. The support vectors are the data points closest to the decision boundary. SVMs work well in high-dimensional spaces and are effective when the number of features exceeds the number of samples. The kernel trick allows SVMs to handle non-linearly separable data by mapping inputs into higher-dimensional spaces using kernels like RBF, polynomial, and sigmoid.

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a new coordinate system where the axes are the principal components — directions of maximum variance. PCA is used for visualization, noise reduction, and speeding up downstream ML algorithms. The number of components is chosen based on the explained variance ratio, typically retaining 95% of total variance.

K-Nearest Neighbors (KNN) is a non-parametric algorithm that classifies a sample based on the majority class of its k nearest neighbors in feature space. Distance metrics include Euclidean, Manhattan, and Minkowski. KNN has no training phase but is slow at inference time for large datasets. The choice of k affects bias-variance tradeoff: small k = high variance, large k = high bias.

Decision Trees split data recursively based on feature thresholds that maximize information gain (entropy) or minimize Gini impurity. Trees are interpretable but prone to overfitting. Pruning techniques like max_depth, min_samples_split, and min_samples_leaf control tree complexity. Random forests and gradient boosting use ensembles of trees to reduce variance and improve generalization.

Naive Bayes classifiers apply Bayes theorem with the naive assumption of conditional independence between features. Despite this simplification, Naive Bayes works well for text classification, spam filtering, and sentiment analysis. Gaussian Naive Bayes assumes continuous features follow a normal distribution. Multinomial Naive Bayes is suited for discrete count data like word frequencies.

Cross-entropy loss (log loss) measures the difference between predicted probability distributions and true labels. It penalizes confident wrong predictions heavily. For binary classification: L = -[y*log(p) + (1-y)*log(1-p)]. For multi-class: L = -sum(y_i * log(p_i)). Neural networks trained for classification typically use cross-entropy with softmax output activation.

Gradient boosting builds an additive model by fitting successive trees to the residuals of the previous model. The learning rate (shrinkage) controls the contribution of each tree. XGBoost adds L1/L2 regularization, column subsampling, and approximate tree learning for speed. LightGBM uses histogram-based splits for further speedup. CatBoost handles categorical features natively without preprocessing.

Stochastic gradient descent updates parameters using gradients computed on small random batches. This introduces noise which can help escape local minima. Key hyperparameters: learning rate, momentum, weight decay. Learning rate warmup starts training at a low rate then increases to the target rate over a fixed number of steps. Gradient clipping prevents exploding gradients by capping gradient norm.

Feature scaling is critical for distance-based algorithms (KNN, SVM) and gradient-based optimization. Min-max normalization maps features to [0,1]. Standardization (z-score) scales to zero mean and unit variance. Robust scaling uses median and IQR, making it resistant to outliers. Tree-based models (Random Forest, XGBoost) are invariant to monotonic feature transformations and do not require scaling.

Regularization prevents overfitting by adding a penalty term to the loss function. L1 regularization (Lasso) adds sum of absolute weights, driving some weights to exactly zero — useful for feature selection. L2 regularization (Ridge) adds sum of squared weights, shrinking all weights toward zero without sparsity. Elastic net combines both L1 and L2 penalties with a mixing parameter alpha.

Precision-Recall curves are more informative than ROC curves for imbalanced datasets where the positive class is rare. The area under the PR curve (AUCPR) summarizes model performance across all classification thresholds. A random classifier's AUCPR equals the prevalence of the positive class, unlike ROC-AUC which is always 0.5 for random classifiers.

SHAP (SHapley Additive exPlanations) values provide consistent and locally accurate feature attributions by averaging over all possible feature coalitions. SHAP explains individual predictions and enables global model interpretation. TreeSHAP computes exact SHAP values for tree-based models in polynomial time. SHAP beeswarm plots show the distribution of feature impacts across all predictions.
""".strip()

with open("ml_concepts.txt", "w", encoding="utf-8") as f:
    f.write(ml_txt)
print("Created: ml_concepts.txt")

# ── File 2: Deep Learning CSV ─────────────────────────────────────────────────
dl_rows = [
    ["topic", "content"],
    ["Batch Normalization", "Batch normalization normalizes layer inputs by subtracting batch mean and dividing by batch standard deviation, then applying learnable scale and shift parameters. It stabilizes training, allows higher learning rates, and acts as a regularizer. Applied before or after the activation function. Layer normalization normalizes across the feature dimension rather than the batch dimension, making it suitable for variable-length sequences in transformers."],
    ["Dropout", "Dropout randomly sets activations to zero during training with probability p, forcing the network to learn redundant representations. At inference time, all neurons are active and outputs are scaled by (1-p). Dropout rates of 0.1-0.5 are common. Spatial dropout drops entire feature maps for CNNs. Variational dropout ties dropout masks across time steps for RNNs. Monte Carlo dropout enables uncertainty estimation at inference."],
    ["Residual Networks", "ResNet introduced skip connections that add the input of a layer to its output: y = F(x) + x. This allows gradients to flow directly through the network during backpropagation, enabling training of very deep networks (50, 101, 152 layers). The residual block learns the residual mapping F(x) = H(x) - x rather than the full mapping H(x). ResNet won ImageNet 2015 and is foundational to modern vision architectures."],
    ["Attention Mechanism", "Attention computes a weighted sum of values V based on the compatibility between queries Q and keys K. Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V. The scaling factor sqrt(d_k) prevents the dot products from growing too large in high dimensions. Multi-head attention projects Q, K, V into multiple subspaces and computes attention in each, enabling the model to focus on different aspects of the input simultaneously."],
    ["BERT", "BERT (Bidirectional Encoder Representations from Transformers) pretrains on masked language modeling (MLM) and next sentence prediction (NSP). MLM masks 15% of tokens and predicts them using both left and right context. BERT uses WordPiece tokenization with a vocabulary of 30,000 tokens. Fine-tuning adds a task-specific head and updates all parameters. BERT-base has 12 layers, 768 hidden dims, 12 attention heads, 110M parameters."],
    ["GPT Architecture", "GPT uses a decoder-only transformer with causal (autoregressive) self-attention, meaning each token can only attend to previous tokens. Pretraining uses next-token prediction on large text corpora. GPT-2 demonstrated that scaling improves zero-shot performance across diverse tasks. GPT-3 (175B parameters) showed in-context learning: the model can perform tasks from just a few examples in the prompt without weight updates."],
    ["Adam Optimizer", "Adam (Adaptive Moment Estimation) maintains per-parameter learning rates using first moment (mean) and second moment (uncentered variance) estimates. Bias correction compensates for initialization at zero. Hyperparameters: alpha (learning rate), beta1 (0.9), beta2 (0.999), epsilon (1e-8). AdamW decouples weight decay from the gradient update, preventing the optimizer from scaling the decay with the adaptive learning rate."],
    ["Convolutional Layers", "A convolutional layer applies learned filters (kernels) across the spatial dimensions of the input using sliding window operations. The output feature map size: (W - F + 2P) / S + 1 where W=input size, F=filter size, P=padding, S=stride. Depthwise separable convolutions split standard convolutions into a depthwise spatial convolution followed by a pointwise 1x1 convolution, reducing parameters by a factor of (1/N + 1/F^2) where N=output channels."],
    ["Generative Adversarial Networks", "GANs consist of a generator G that maps noise z to fake data and a discriminator D that distinguishes real from fake. Training alternates between maximizing D's accuracy and minimizing it from G's perspective. Mode collapse occurs when G produces limited diversity. Wasserstein GANs use Earth Mover distance and gradient penalty for more stable training. StyleGAN introduces adaptive instance normalization (AdaIN) for fine-grained style control."],
    ["Variational Autoencoders", "VAEs learn a probabilistic latent space by encoding inputs as distributions (mean and variance) rather than point estimates. The reparameterization trick enables backpropagation through sampling. The ELBO loss combines reconstruction loss with KL divergence from the prior N(0,I). VAEs enable smooth interpolation between data points in latent space. Beta-VAEs increase the KL weight to encourage disentangled representations."],
    ["Transfer Learning Fine-tuning", "Fine-tuning updates pretrained model weights on a target task dataset. Common strategies: freeze all except the classification head (feature extraction), unfreeze top layers only (partial fine-tuning), or unfreeze all layers (full fine-tuning). Use a lower learning rate for pretrained layers (e.g., 1e-5) vs. the new head (e.g., 1e-3). Gradual unfreezing starts with the top layer and progressively unfreezes lower layers to avoid catastrophic forgetting."],
    ["Neural Architecture Search", "NAS automates the design of neural network architectures by searching over architecture space using reinforcement learning, evolutionary algorithms, or gradient-based methods. DARTS (Differentiable Architecture Search) relaxes discrete architecture choices to continuous by using softmax over candidate operations. EfficientNet was discovered via NAS combined with compound scaling that uniformly scales depth, width, and resolution."],
]

with open("deep_learning_notes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(dl_rows)
print("Created: deep_learning_notes.csv")

# ── File 3: RAG Systems Guide TXT ─────────────────────────────────────────────
rag_txt = """
Dense Passage Retrieval (DPR) trains dual-encoder models to embed questions and passages into a shared dense vector space where relevant pairs have high dot product similarity. The question encoder and passage encoder are trained contrastively using in-batch negatives. DPR outperforms BM25 for open-domain question answering. Pre-computed passage embeddings are stored in FAISS for efficient approximate nearest neighbor retrieval.

Sentence transformers fine-tune transformer models using siamese or triplet network architectures to produce semantically meaningful sentence embeddings. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings with strong performance and fast inference. Models are trained on natural language inference, paraphrase, and semantic textual similarity datasets. Mean pooling over token embeddings is preferred over CLS token representations.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. IndexFlatL2 performs exact brute-force search. IndexIVFFlat partitions vectors into Voronoi cells and searches only nearby cells. IndexHNSWFlat uses hierarchical navigable small world graphs for sub-linear search with high recall. IndexPQ uses product quantization to compress vectors and reduce memory by 8-32x.

Re-ranking with cross-encoders improves retrieval precision by jointly encoding the query and each candidate document and scoring their relevance. Cross-encoders are slower than bi-encoders but more accurate because they model query-document interactions. They are used as a second stage after initial retrieval narrows candidates from thousands to tens. MS-MARCO trained cross-encoders are commonly used for general-purpose re-ranking.

Contextual compression filters retrieved documents to keep only the most relevant passages before passing context to the LLM. The LLM Embeddings Filter, LLMChainFilter, and EmbeddingsRedundantFilter are compression strategies. This reduces hallucination by removing irrelevant context that might distract the model. Cohere Rerank API provides a hosted re-ranking service using their proprietary model.

Self-RAG trains an LLM to actively decide when to retrieve, evaluate retrieved passages, and critique its own outputs using special reflection tokens. The model generates retrieval tokens (Retrieve, No Retrieve), relevance tokens (Relevant, Irrelevant), and support tokens (Fully supported, Partially supported, No support). This adaptive retrieval reduces unnecessary API calls and improves factual accuracy compared to always-retrieve RAG.

Corrective RAG (CRAG) evaluates retrieved documents and triggers a web search when retrieval quality is low. A lightweight evaluator scores each retrieved document as Correct, Incorrect, or Ambiguous. Documents scored Incorrect trigger reformulated web searches using Tavily or SerpAPI. Retrieved web results are processed with a knowledge refinement step that decomposes and filters the information before augmenting the LLM context.

Graph RAG structures knowledge as a graph where nodes are entities and edges are relationships extracted from source documents. Entity extraction uses LLMs to identify entities and relationships. Community detection algorithms identify thematically related entity clusters. Queries are answered using both local entity retrieval and global community summaries. Microsoft's GraphRAG achieves strong performance on questions requiring synthesis across multiple documents.

Hypothetical Document Embeddings (HyDE) generates a hypothetical answer to the query using an LLM, then embeds the hypothetical answer for retrieval instead of the original query. The hypothetical answer is typically more similar to relevant documents in embedding space than the query itself. HyDE improves recall for zero-shot retrieval tasks where queries and documents have different linguistic styles.

Multi-vector retrieval stores multiple embedding representations per document: summary embedding for coarse retrieval, chunk embeddings for precise matching, and parent document IDs for context expansion. The parent document retriever uses child chunks for retrieval but returns the full parent document as context. This balances retrieval precision with context richness for the LLM.

RAG evaluation with RAGAS measures: context_precision (fraction of retrieved context that is relevant), context_recall (fraction of ground truth covered by retrieved context), faithfulness (proportion of response claims supported by context), and answer_relevancy (how well the answer addresses the question). These metrics enable automated evaluation without human annotation using an LLM-as-judge scoring approach.

Streaming RAG passes retrieved context and the user query to the LLM and streams the response token by token to the frontend using server-sent events (SSE). This reduces perceived latency significantly. LangChain and LlamaIndex both support streaming callbacks. FastAPI with StreamingResponse enables efficient server-side streaming. The retrieval step is not streamed but pre-computation of embeddings and approximate search keeps it under 100ms.

Long-context RAG handles documents that exceed the context window by using map-reduce summarization: each chunk is summarized separately, then summaries are synthesized into a final answer. Alternatively, LLMs with 128k+ context windows (Claude, Gemini) can process many retrieved chunks directly. Lost-in-the-middle research shows LLMs attend better to information at the start and end of context, so retrieved chunks should be ordered by relevance with the most relevant at the top.
""".strip()

with open("rag_systems_guide.txt", "w", encoding="utf-8") as f:
    f.write(rag_txt)
print("Created: rag_systems_guide.txt")

print("\nAll 3 files created successfully!")
print("Upload them in the PulseRAG 'Upload Docs' tab.")
print("\nFiles:")
for fname in ["ml_concepts.txt", "deep_learning_notes.csv", "rag_systems_guide.txt"]:
    size = os.path.getsize(fname)
    print(f"  {fname:35s} {size:,} bytes")