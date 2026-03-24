# Report — HW2

## 1. Objective

This work studies two compact transformer-based computer vision pipelines:

- **Variant A** — token-based image classification with a small Vision Transformer
- **Variant B** — query-based prediction of object boxes and masks

The main goal was not to build the strongest possible model, but to reproduce the core behavior of token and query architectures in a lightweight experimental setup.

---

## 2. Variant A — Vision Transformer Classification

### 2.1 Experimental setup

For Variant A, I trained compact ViT models on **CIFAR-10** with input size `64×64`.

Two configurations were compared:

- `vit_p8_d2_dim128_cls`
- `vit_p16_d3_dim192_cls`

### 2.2 Patch size and sequence length

For an input image of `64×64`:

#### Configuration A1: patch size = 8
- Number of patches: `(64 / 8)^2 = 64`
- Sequence length with class token: `65`

#### Configuration A2: patch size = 16
- Number of patches: `(64 / 16)^2 = 16`
- Sequence length with class token: `17`

This difference is important because self-attention cost grows quadratically with sequence length.

### 2.3 Self-attention cost

A simple attention cost proxy per layer is:

\[
O(N^2 \cdot D)
\]

where:
- `N` is sequence length
- `D` is embedding dimension

For the two tested configurations:

- `p=8` → `N=65` → `65^2 = 4225`
- `p=16` → `N=17` → `17^2 = 289`

So even before considering depth and hidden size, the smaller patch model is much more expensive in the attention blocks.

### 2.4 Results

The quantitative comparison is summarized in `results_a.csv`.

The general pattern is:

- smaller patches improve detail and usually improve accuracy
- larger patches reduce sequence length and computational cost
- the cheaper configuration is faster, but loses some fine-grained information

### 2.5 Error analysis

The confusion matrix shows that the model most often fails when:
- classes have similar global structure
- the object is small relative to the background
- local detail is important for distinguishing categories

This explains why the smaller patch size is usually preferable despite the higher attention cost.

### 2.6 Conclusion for Variant A

Variant A confirms the expected ViT trade-off:

- smaller patches → better local representation, longer sequence, higher cost
- larger patches → shorter sequence, lower cost, weaker detail preservation

---

## 3. Variant B — Query-Based Detection and Masks

### 3.1 Experimental setup

Variant B was implemented as a compact query-based prediction model with:
- visual features extracted from the image
- a fixed set of learnable object queries
- cross-attention between queries and image features
- prediction heads for objectness, boxes, and masks

Two model types were compared:
- **box-only**
- **box+mask**

### 3.2 Number of queries

The tested query counts are listed in `results_b.csv`.

In all configurations, the number of queries is fixed before inference.  
Each query is intended to represent one potential object hypothesis.

### 3.3 Main tensor shapes

The main tensors in the model can be described as follows:

#### Visual feature tokens
\[
B \times T \times D
\]

where:
- `B` = batch size
- `T` = number of spatial tokens
- `D` = hidden dimension

#### Object queries
\[
B \times Q \times D
\]

where:
- `Q` = number of learnable object queries

#### Decoder output
\[
B \times Q \times D
\]

Each query is updated by attending to the image features.

#### Box prediction
\[
B \times Q \times 4
\]

#### Objectness / class prediction
\[
B \times Q \times 1
\]
or
\[
B \times Q \times C
\]

depending on the head design.

#### Mask prediction
\[
B \times Q \times H \times W
\]

### 3.4 How a query becomes an object prediction

A query starts as a learnable embedding.  
During decoding:

1. image features are extracted from the input image
2. the query attends to visual tokens through cross-attention
3. the updated query feature becomes an object-level representation
4. this representation is passed into:
   - an objectness/class head
   - a box regression head
   - optionally a mask head
5. the final predictions are filtered by score thresholding and post-processing

This means that each query behaves like an object slot that either stays inactive or produces one detection.

### 3.5 Qualitative analysis

The qualitative outputs show three useful aspects of the model:

- **cross-attention maps** indicate which image regions are used by each query
- **predicted boxes** show whether the query localizes an object correctly
- **predicted masks** show whether the query captures object shape rather than only coarse location

These visualizations help explain both successful detections and common failure cases.

### 3.6 Error analysis

The most common failure modes are:

- duplicate queries predicting the same object
- weak or missing predictions for harder objects
- inaccurate masks near object boundaries
- background-focused queries producing false positives

The mask-enabled model usually provides richer output, but it is also harder to optimize than the simpler box-only setup.

### 3.7 Comparison between box-only and box+mask variants

The comparison in `results_b.csv` shows the following general tendency:

- **box-only** is simpler and often slightly easier to stabilize
- **box+mask** is more expressive and provides richer outputs
- the quality of the final predictions depends strongly on:
  - query count
  - confidence threshold
  - suppression of duplicate predictions

### 3.8 Conclusion for Variant B

Variant B demonstrates how a fixed set of object queries can produce structured outputs from image features.  
The architecture naturally supports both box prediction and mask generation, but the richer mask variant requires more careful training and post-processing.

---

## 4. Overall Discussion

The two variants represent two different uses of transformer-style computation in vision:

- **Variant A** treats the image as a sequence of tokens and predicts a single global label
- **Variant B** uses a fixed set of object queries to produce a set of structured outputs

The main practical observations are:

- tokenization quality depends strongly on patch size
- attention cost grows very quickly with sequence length
- query-based decoding is flexible and interpretable
- cross-attention maps are useful for understanding model behavior
- mask prediction is feasible in a compact setup, but noticeably harder than box regression

---

## 5. Final Conclusion

This project successfully reproduces two important ideas from modern transformer-based computer vision:

1. **token-based visual representation** for image classification
2. **query-based structured prediction** for detection and segmentation

Even in a lightweight setup, both variants reveal the central trade-offs of these architectures:
- detail versus efficiency in token-based models
- flexibility versus optimization difficulty in query-based models
