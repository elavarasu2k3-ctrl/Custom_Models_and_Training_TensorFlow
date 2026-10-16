# Task — 17: Practical Deep Learning Defaults & Custom Models and Training with TensorFlow

> **Goal:** Finish the deep-network training toolkit with a set of sensible default configurations (and the exceptions to them), then drop below `fit()`: use TensorFlow's low-level API to write custom losses, metrics, layers, models, gradients, and full training loops — and speed them up with TF Functions and graphs.

> **Note:** This continues the ML track (Task 16 → Task 17). Task 16 taught you *what* techniques exist (initialization, Batch Norm, optimizers, schedules, dropout) and how to use them through Keras. Task 17 gives you the practical defaults for combining them, and then the control to go beyond what `compile()` + `fit()` allow. Roughly 95% of real projects only need Keras and `tf.data`; this task is for the cases that don't.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What You Will Learn](#2-what-you-will-learn)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Phase 1 — Default DNN Configurations & Practical Guidelines](#phase-1--default-dnn-configurations--practical-guidelines)
5. [Phase 2 — A Quick Tour of TensorFlow & Using It Like NumPy](#phase-2--a-quick-tour-of-tensorflow--using-it-like-numpy)
6. [Phase 3 — Custom Loss Functions & Saving Custom Components](#phase-3--custom-loss-functions--saving-custom-components)
7. [Phase 4 — Custom Activations, Initializers, Regularizers, Constraints & Metrics](#phase-4--custom-activations-initializers-regularizers-constraints--metrics)
8. [Phase 5 — Custom Layers, Custom Models & Losses Based on Model Internals](#phase-5--custom-layers-custom-models--losses-based-on-model-internals)
9. [Phase 6 — Computing Gradients with Autodiff](#phase-6--computing-gradients-with-autodiff)
10. [Phase 7 — Custom Training Loops](#phase-7--custom-training-loops)
11. [Phase 8 — TensorFlow Functions, Graphs & Capstone](#phase-8--tensorflow-functions-graphs--capstone)
12. [Exercises (Must Do)](#exercises-must-do)
13. [Deliverables Checklist](#deliverables-checklist)
14. [Reference: Key Concepts Glossary](#reference-key-concepts-glossary)
15. [Resources](#resources)

---

## 1. Project Overview

### The Scenario

By the end of Task 16 you could train a deep network well. Real projects, though, eventually hit something Keras doesn't give you out of the box: a loss that isn't in the library, a metric that can't be averaged over batches, a layer from a paper, a regularizer based on a hidden activation, or a training algorithm with two optimizers for two parts of the network (exactly what the Wide & Deep paper in Task 14 did). Task 17 starts with a decision framework — which defaults to use, and when you're in an exception case — and then walks through the whole customization stack, from tensors up to graph-compiled training loops.

### Why This Matters

| Concept | Where It Shows Up in Practice |
|---|---|
| Default DNN configurations | A strong starting point that needs little tuning, and a clear list of when to deviate (sparse, low-latency, risk-sensitive models) |
| TensorFlow as "NumPy + GPU + autodiff" | Understanding dtypes, immutable tensors vs. `tf.Variable`, and the other data structures (sparse, ragged, string) |
| Custom losses (Huber) | Noisy data where MSE over-penalizes outliers and MAE converges slowly |
| Streaming metrics | Correct precision/recall tracking across batches (averaging per-batch precision is *wrong*) |
| Custom layers/models | Reusable blocks (residual blocks, layer norm) and architectures with skip connections |
| `GradientTape` & custom gradients | Anything that needs gradients of arbitrary computations, including numerically stable custom derivatives |
| Custom training loops | Multiple optimizers, gradient transformations beyond clipping, full control and transparency |
| TF Functions / AutoGraph | The speed-up and portability that turn custom Python code into optimized graphs |

---

## 2. What You Will Learn

By completing this task, you will be able to:

- Apply the default DNN configuration (He init + ELU, BN if deep, early stopping, momentum/RMSProp/Nadam, 1cycle) and the self-normalizing variant (LeCun + SELU, no BN, alpha dropout if needed)
- Recognize and handle the exceptions: sparse models (ℓ1 regularization, TF Model Optimization Toolkit), low-latency models (fewer layers, fold BN into previous layers, leaky ReLU/ReLU, lower precision), and risk-sensitive models (MC Dropout)
- Describe TensorFlow's core features: NumPy-like API with GPU support, distributed computing, a JIT-style graph compiler, portable graph export, and autodiff with built-in optimizers
- Create and manipulate tensors (`tf.constant`, indexing, operations, `@` for `tf.matmul`) and explain why TF doesn't auto-convert types and why you must use `tf.cast`
- Explain why tensors are immutable and when to use `tf.Variable` and its `assign()` methods
- Name the other TF data structures: sparse tensors, tensor arrays, ragged tensors, string tensors, sets, and queues
- Write a custom loss as a function (Huber), as a configured closure, and as a `keras.losses.Loss` subclass with `get_config()`, and load saved models with `custom_objects`
- Write custom activations, initializers, regularizers, and constraints, and say when subclassing is needed (to persist hyperparameters)
- Explain the difference between losses and metrics, and implement a streaming (stateful) metric via `keras.metrics.Metric`
- Build custom layers: stateless (`Lambda`), stateful (`build()`, `call()`, `add_weight()`), multi-input/output, and training/inference-aware (`training` argument)
- Build a custom model (a residual regressor with reusable `ResidualBlock` layers) and explain when to subclass `Layer` vs. `Model`
- Add losses and metrics based on model internals with `add_loss()` and `add_metric()` (e.g., a reconstruction loss)
- Compute gradients with `tf.GradientTape`, and use persistent tapes, `tape.watch()`, `jacobian()`, `tf.stop_gradient()`, and `@tf.custom_gradient`
- Write a complete custom training loop (nested epoch/batch loops, tape, `tf.add_n` of losses, `apply_gradients`, metric tracking, constraint application, training-mode handling)
- Convert Python functions to TF Functions, explain AutoGraph and tracing, and follow the TF Function rules (use TF ops not NumPy, `tf.range` loops, create variables only once, etc.)

---

## 3. Prerequisites & Setup

### Required Knowledge

- Task 16 (Keras APIs, initialization, activations, Batch Norm, optimizers, schedules, regularization, MC Dropout)
- Task 7 (Gradient Descent, regularization), Task 14 (Wide & Deep, two-tower models — motivation for custom loops)

### Environment Setup

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib --break-system-packages
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

> **Version note:** the source material targets TensorFlow 2.0-era `tf.keras`. In current releases (Keras 3) a few details differ: use `learning_rate=` instead of `lr=`, metrics expose `reset_state()` instead of `reset_states()`, and some legacy methods such as `add_metric()` and `compute_output_shape()` requirements have changed or been removed. Where a snippet below fails on your version, check the Keras migration notes (or install `tf-keras` to reproduce the older behavior exactly), and record the fix in your notebook.

---

## Phase 1 — Default DNN Configurations & Practical Guidelines



### Tasks to Perform

1. Write the **default DNN configuration** from memory, then check it against the table below. It is a starting point that works in most cases with little tuning — not a hard rule.
2. Write the **self-normalizing net** configuration. State when it applies: a simple stack of dense layers.
3. State the always-on reminders: **normalize the input features**; try to reuse parts of a pretrained network for a similar problem; use unsupervised pretraining if you have lots of unlabeled data; use pretraining on an auxiliary task if you have lots of labeled data for a similar task.
4. List the three exceptions and the response to each:
   - **Sparse model:** ℓ1 regularization (optionally zero out tiny weights after training); for even sparser, the TensorFlow Model Optimization Toolkit. This breaks self-normalization, so use the default config instead of the SELU one.
   - **Low-latency model:** fewer layers, fold Batch Norm layers into the previous layers, a faster activation (leaky ReLU or ReLU), a sparse model, and lower float precision (32 → 16 or 8 bits).
   - **Risk-sensitive application or latency isn't critical:** use MC Dropout for better accuracy, more reliable probabilities, and uncertainty estimates.
5. Answer the conceptual review questions (write 2–4 sentences each):
   - Is it OK to initialize all weights to the same value, as long as it's randomly chosen with He initialization?
   - Is it OK to initialize bias terms to 0?
   - Name three advantages of SELU over ReLU.
   - When would you use SELU, leaky ReLU (and variants), ReLU, tanh, logistic, and softmax?
   - What may happen if SGD momentum is set too close to 1 (e.g., 0.99999)?
   - Name three ways to produce a sparse model.
   - Does dropout slow down training? Inference? What about MC Dropout?
6. Mark each of these on the design decision tree: "I need a custom loss," "I need a custom training algorithm," "I need to tweak gradients beyond clipping," "I need multiple optimizers" — and note that these are the cases that require the lower-level API in the rest of this task.

### 1.1 Default Configuration Tables

| Hyperparameter | Default DNN | Self-normalizing net |
|---|---|---|
| Kernel initializer | He | LeCun |
| Activation | ELU | SELU |
| Normalization | None if shallow; Batch Norm if deep | None (self-normalization) |
| Regularization | Early stopping (+ ℓ2 if needed) | Alpha dropout if needed |
| Optimizer | Momentum (or RMSProp or Nadam) | Momentum (or RMSProp or Nadam) |
| Learning rate schedule | 1cycle | 1cycle |

**Deliverable for Phase 1:** Both tables from memory, the three-exceptions summary, and written answers to the seven review questions.

---

## Phase 2 — A Quick Tour of TensorFlow & Using It Like NumPy



### Tasks to Perform

1. Summarize what TensorFlow offers: a NumPy-like core with GPU support, distributed computing, a JIT-style compiler that extracts and optimizes a computation graph from Python functions, portable graph export (train in Python, run elsewhere), and autodiff with good optimizers (RMSProp, Nadam…).
2. Name the ecosystem pieces: `tf.keras`, `tf.data`, `tf.io`, `tf.image`, `tf.signal`, TensorBoard, TFX, TF Serving, TensorFlow Hub, the model garden, TensorFlow Lite, TensorFlow.js.
3. Explain **ops and kernels:** each op has a C++ implementation with device-specific kernels (CPU, GPU, TPU).
4. Create tensors: `tf.constant([[1., 2., 3.], [4., 5., 6.]])` (matrix) and `tf.constant(42)` (scalar); inspect `.shape` and `.dtype`. Practice indexing (`t[:, 1:]`, `t[..., 1, tf.newaxis]`).
5. Run operations: `t + 10`, `tf.square(t)`, `t @ tf.transpose(t)`. Note `t + 10` ≡ `tf.add(t, 10)` and `@` ≡ `tf.matmul`.
6. Explain the NumPy-name differences: `tf.reduce_mean/sum/max`, `tf.math.log`, and why `tf.transpose(t)` creates a copy while NumPy's `t.T` is a view; why `reduce_sum` can give slightly different results across runs on GPU (order of additions not guaranteed with 32-bit floats).
7. Convert between NumPy and TF: `tf.constant(np_array)`, `t.numpy()`, `np.square(t)`, `tf.square(a)`. Note NumPy defaults to 64-bit floats and TF to 32-bit; set `dtype=tf.float32` when converting.
8. Demonstrate **no automatic type conversion:** `tf.constant(2.) + tf.constant(40)` raises an error; use `tf.cast(...)`. Explain why (hidden conversions hurt performance).
9. Create a `tf.Variable`; modify it in place with `assign()`, `assign_add()`, slice assignment (`v[0, 1].assign(42)`), and `scatter_nd_update()`. Explain why plain tensors can't be used as weights, and that you'll rarely create variables by hand (Keras `add_weight()`).
10. List the other data structures with a one-line use case each: **sparse tensors**, **tensor arrays**, **ragged tensors**, **string tensors** (byte strings vs. Unicode code points), **sets**, **queues** (FIFO, priority, random shuffle, padding FIFO).

**Deliverable for Phase 2:** A short notebook demonstrating each of tasks 4–9 with printed outputs, plus a one-line description for each of the six extra data structures.

---

## Phase 3 — Custom Loss Functions & Saving Custom Components



### Tasks to Perform

1. Explain why you might want **Huber loss** for noisy regression data: MSE over-penalizes large errors; MAE doesn't converge as precisely; Huber is quadratic for small errors and linear for large ones.
2. Implement it as a function using only TF ops:
   ```python
   def huber_fn(y_true, y_pred):
       error = y_true - y_pred
       is_small_error = tf.abs(error) < 1
       squared_loss = tf.square(error) / 2
       linear_loss = tf.abs(error) - 0.5
       return tf.where(is_small_error, squared_loss, linear_loss)
   ```
   Explain why it should be **vectorized** and return **one loss per instance** (so Keras can apply sample/class weights).
3. Compile and train a regression model on California Housing with `loss=huber_fn`; compare with `keras.losses.Huber()` and with MSE.
4. Save the model, then load it back with `custom_objects={"huber_fn": huber_fn}`. Explain why this mapping is needed (Keras saves the *name*).
5. Create a configurable version with a closure `create_huber(threshold)`. Show that the threshold is **not** saved with the model, so you must pass `custom_objects={"huber_fn": create_huber(2.0)}` when loading.
6. Fix this by subclassing `keras.losses.Loss`: `__init__(threshold, **kwargs)`, `call(y_true, y_pred)`, and `get_config()` that merges the base config and `threshold`. Explain the `reduction` argument (`"sum_over_batch_size"` by default, `"sum"`, `"none"`) and why it divides by batch size rather than the sum of weights.
7. Save/load again with `custom_objects={"HuberLoss": HuberLoss}` and verify the threshold survives. Explain what `from_config()` does on load.
8. Note which components Keras officially supports subclassing for (layers, models, callbacks, regularizers) vs. those where subclassing may not be portable to other Keras implementations (losses, metrics, initializers, constraints).

### 3.1 Loss as Class

```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**Deliverable for Phase 3:** Three working Huber variants (function, closure, subclass), a save→load round trip for each showing exactly what is and isn't persisted, and a comparison of test error vs. built-in MSE and `keras.losses.Huber`.

---

## Phase 4 — Custom Activations, Initializers, Regularizers, Constraints & Metrics



### Tasks to Perform

1. Implement the four simple function-style components and verify each against its built-in equivalent:
   ```python
   def my_softplus(z):                      # tf.nn.softplus
       return tf.math.log(tf.exp(z) + 1.0)

   def my_glorot_initializer(shape, dtype=tf.float32):   # glorot_normal
       stddev = tf.sqrt(2. / (shape[0] + shape[1]))
       return tf.random.normal(shape, stddev=stddev, dtype=dtype)

   def my_l1_regularizer(weights):          # regularizers.l1(0.01)
       return tf.reduce_sum(tf.abs(0.01 * weights))

   def my_positive_weights(weights):        # constraints.nonneg()
       return tf.where(weights < 0., tf.zeros_like(weights), weights)
   ```
2. Plug all four into a single `Dense` layer and explain *when each is called*: the activation on the layer's output; the initializer at weight creation; the regularizer at each step (its value is added to the loss); the constraint after each training step (weights replaced by the constrained weights).
3. Subclass `keras.regularizers.Regularizer` to create `MyL1Regularizer(factor)` that persists `factor` through `get_config()`. State the method you must implement for each family: `call()` for losses/layers/models, `__call__()` for regularizers/initializers/constraints.
4. State the conceptual difference between **losses** and **metrics**: losses must be differentiable (usable by Gradient Descent) and need not be human-interpretable; metrics evaluate the model, must be interpretable, and may be non-differentiable.
5. Use a function as a metric (`metrics=[create_huber(2.0)]`) and note Keras averages it over batches.
6. Demonstrate why averaging is **wrong** for precision: batch 1 → 4/5 correct (80%), batch 2 → 0/3 correct (0%); the naive mean is 40%, the true precision is 4/8 = 50%. Reproduce this with `keras.metrics.Precision()` (returns 0.8 then 0.5), and inspect `.result()`, `.variables` (true/false positives), and `.reset_states()`.
7. Implement a **streaming metric** subclassing `keras.metrics.Metric`: `add_weight()` for `total` and `count`, `update_state(y_true, y_pred, sample_weight=None)`, `result()`, and `get_config()`. Then re-implement it more simply by subclassing `keras.metrics.Mean`.

### 4.1 Streaming Metric

```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

**Deliverable for Phase 4:** A table verifying each custom function against its built-in equivalent, the 80%/0%/40%/50% precision demonstration, and a working streaming metric (both the `Metric` and the `Mean`-subclass versions).

---

## Phase 5 — Custom Layers, Custom Models & Losses Based on Model Internals



### Tasks to Perform

1. Build a **weightless custom layer** with `keras.layers.Lambda(lambda x: tf.exp(x))`. Explain its use as an output layer for regression targets spanning very different scales (e.g., 0.001, 10, 1,000).
2. Build a **stateful layer** by subclassing `keras.layers.Layer`: reimplement a simplified `Dense` as `MyDense` with:
   - `__init__(units, activation=None, **kwargs)` — pass `**kwargs` to the parent; convert activation via `keras.activations.get()`
   - `build(batch_input_shape)` — create `kernel` and `bias` with `add_weight()`; call `super().build()` **at the end**
   - `call(X)` — `self.activation(X @ self.kernel + self.bias)`
   - `compute_output_shape()` (usually optional in `tf.keras`) and `get_config()` (use `keras.activations.serialize`)
3. Use `MyDense` in a model and confirm it trains and saves/loads (with `custom_objects`).
4. Build a **multi-input, multi-output** toy layer (`MyMultiLayer`: two inputs → three outputs); note it only works with the Functional and Subclassing APIs, not Sequential.
5. Build a **training-aware layer** (`MyGaussianNoise`): add noise only when `training=True`; compare with `keras.layers.GaussianNoise`.
6. Build the custom model from the chapter's figure: a `ResidualBlock` layer (n dense ELU layers + skip connection `inputs + Z`) and a `ResidualRegressor` model (dense → block1 ×4 → block2 → output). Explain that Keras auto-tracks layers stored in a list attribute.
7. State the rule: **components → subclass `Layer`; the thing you train → subclass `Model`** (a `Model` is a `Layer` plus `compile/fit/evaluate/predict/save`). Implement `get_config()` in both classes if you want `save()`/`load_model()`; otherwise use `save_weights()`/`load_weights()`.
8. Add a **loss based on model internals** with `add_loss()`: build `ReconstructingRegressor` (5 SELU hidden layers + an auxiliary reconstruction layer created in `build()` because its width equals the unknown input size). In `call()`, compute `recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))` and `self.add_loss(0.05 * recon_loss)`. Explain why the factor 0.05 keeps the auxiliary loss from dominating, and that it acts as a regularizer.
9. Add a **metric based on internals** (a `keras.metrics.Mean` object for the reconstruction error, registered with `add_metric()`) and verify that both the total loss and the reconstruction error fall during training. (If your Keras version removed `add_metric`, achieve the same by tracking the metric in a custom training loop — see Phase 7.)

**Deliverable for Phase 5:** Working `MyDense`, `MyGaussianNoise`, `ResidualBlock`/`ResidualRegressor`, and `ReconstructingRegressor`, each trained on California Housing, with a one-line comparison of test error vs. an ordinary MLP.

---

## Phase 6 — Computing Gradients with Autodiff



### Tasks to Perform

1. Take `f(w1, w2) = 3·w1² + 2·w1·w2`. Derive the partial derivatives by hand (`6w1 + 2w2` and `2w1`) and evaluate at `(5, 3)` → `(36, 10)`.
2. Approximate them with **finite differences** (`eps = 1e-6`) and note two drawbacks: it's only approximate, and it needs at least one function call **per parameter** — intractable for big networks.
3. Compute them with TF: wrap `f` in `with tf.GradientTape() as tape:` using `tf.Variable` inputs, then `tape.gradient(z, [w1, w2])`. Confirm `[36., 10.]` and explain why this needs only one forward and one reverse pass regardless of the number of variables.
4. Demonstrate that `gradient()` can only be called **once** per tape (second call raises `RuntimeError`); fix it with `persistent=True` and `del tape` afterward.
5. Show that tapes only track **variables**: gradients w.r.t. constants return `[None, None]`; force tracking with `tape.watch(c1)`. Describe a use case (a regularizer penalizing activations that change a lot when inputs change slightly — based on the gradient of activations w.r.t. inputs).
6. Explain two efficiency tips: keep only the minimum inside the tape, and pause recording with `tape.stop_recording()`.
7. Explain what happens when you take the gradient of a **vector**: TF differentiates its **sum**; use `tape.jacobian()` for per-loss gradients. Note second-order derivatives (Hessians) are possible but rarely needed.
8. Use `tf.stop_gradient()` on `2*w1*w2` and verify `gradients == [30., None]` (forward result unchanged, but no gradient flows through that term).
9. Reproduce the **numerical-stability** problem: gradients of `my_softplus` at `x = 100` give `NaN` (inf/inf). Fix it with `@tf.custom_gradient`:
   ```python
   @tf.custom_gradient
   def my_better_softplus(z):
       exp = tf.exp(z)
       def my_softplus_gradients(grad):
           return grad / (1 + 1 / exp)
       return tf.math.log(exp + 1), my_softplus_gradients
   ```
   Explain the chain-rule role of the incoming `grad`, and that the forward output can still overflow for large inputs (workaround: `tf.where` to return the input when it's large).

**Deliverable for Phase 6:** The hand-derived, finite-difference, and `GradientTape` gradients side by side; demonstrations of persistent tapes, `watch`, `stop_gradient` (`[30., None]`), and the `NaN` → fixed custom-gradient example.

---

## Phase 7 — Custom Training Loops



### Tasks to Perform

1. State when you need a custom loop (and when you don't): `fit()` is enough unless you need e.g. **two different optimizers** (the Wide & Deep paper: one for the wide path, one for the deep path — see Task 14), gradient transformations beyond clipping, or you simply want full transparency. Custom loops are longer, more error-prone, and harder to maintain, so prefer `fit()` in a team unless you need the flexibility.
2. Build a small model (don't compile it): Dense(30, ELU, He init, ℓ2(0.05)) → Dense(1, ℓ2(0.05)). Write a `random_batch(X, y, batch_size=32)` helper and a `print_status_bar(...)` helper.
3. Set hyperparameters: `n_epochs=5`, `batch_size=32`, `n_steps = len(X_train)//batch_size`, `Nadam(learning_rate=0.01)`, `loss_fn = keras.losses.mean_squared_error`, a `keras.metrics.Mean()` for loss, and `MeanAbsoluteError()` as the extra metric.
4. Write the loop: nested epoch/step loops; inside `tf.GradientTape()`:
   - `y_pred = model(X_batch, training=True)`
   - `main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))` (reduce here; this is where you'd apply per-instance weights)
   - `loss = tf.add_n([main_loss] + model.losses)` (add the regularization losses)
   - `gradients = tape.gradient(loss, model.trainable_variables)` (trainable variables only, not all variables)
   - `optimizer.apply_gradients(zip(gradients, model.trainable_variables))`
   - update the mean loss and metrics, print status; at epoch end, print a final status line and call `reset_states()` on all metrics
5. Explain why `clipnorm`/`clipvalue` set on the optimizer still work, and where to put any **other gradient transformation** (between `tape.gradient` and `apply_gradients`).
6. If your layers have **constraints**, apply them right after `apply_gradients`:
   ```python
   for variable in model.variables:
       if variable.constraint is not None:
           variable.assign(variable.constraint(variable))
   ```
7. List the things this loop does **not** handle for free: training/inference-mode layers (BatchNorm, Dropout — pass `training=True` and propagate it), validation evaluation, callbacks, and learning-rate schedules. Add a validation pass at the end of each epoch.
8. Note the sampling caveat: random batches mean some instances are processed more than once and some not at all in an epoch — fine in practice, but a `tf.data` pipeline (next topic in the track) is the better approach.
9. Extend the loop to use **two optimizers**: a different optimizer and learning rate for the upper layers vs. the lower layers. Verify with a learning-curve plot that both groups of variables are updating.

**Deliverable for Phase 7:** A working custom loop with metrics, validation, constraint handling, and a two-optimizer variant, plus a short list of every `fit()` feature you had to re-implement by hand.

---

## Phase 8 — TensorFlow Functions, Graphs & Capstone



### Tasks to Perform

1. Convert `cube(x) = x ** 3` to a TF Function with `tf.function(cube)` and as a `@tf.function` decorator; compare outputs for Python ints and tensors. Access the original with `.python_function`.
2. Explain what TensorFlow does with the graph: prune unused nodes, simplify expressions (e.g., `1 + 2` → `3`), run independent ops in parallel. Note the trivial `cube` is actually *slower* as a TF Function (nothing to optimize).
3. State that Keras auto-converts your custom losses/metrics/layers to TF Functions — most of this is transparent. To disable, use `dynamic=True` on a layer/model or `run_eagerly=True` in `compile()`.
4. Explain **polymorphism and graph caching:** a new graph is traced per unique input *shape and dtype* for tensor arguments, but per unique *value* for Python numbers — so many distinct Python-number arguments create many graphs and waste RAM. Reserve Python values for arguments with few unique values (like hyperparameters).
5. Explain **AutoGraph** (analyzes source code and replaces `for`/`while`/`if` with `tf.while_loop`/`tf.cond`) and **tracing** (runs the upgraded function with symbolic tensors in graph mode, adding nodes instead of computing). View the generated code with `tf.autograph.to_code(fn.python_function)`.
6. Write and verify the **TF Function rules** with a tiny demonstration of each:
   - external-library calls (NumPy, stdlib) run only during tracing — show `np.random.rand()` returning the same value vs. `tf.random.uniform([])` returning a new value each call
   - Python side effects (logging, counters) happen only during tracing
   - `tf.py_function()` wraps arbitrary Python but hurts performance and portability
   - variables must be created only on the first call (prefer `build()`); use `assign()`, not `=`
   - source code must be available
   - only loops over tensors/datasets are captured — use `for i in tf.range(x)`, not `for i in range(x)`
   - prefer vectorized implementations
7. Decorate your Phase 7 training step with `@tf.function` and time epochs with and without it; report the speed-up.
8. **Capstone A — Layer Normalization layer:** implement a custom layer with trainable `α` (init 1s) and `β` (init 0s) of shape `input_shape[-1:]`. In `call()`, compute `μ, σ² = tf.nn.moments(inputs, axes=-1, keepdims=True)`, then return `α ⊗ (X − μ)/(√σ² + ε) + β` with `ε = 0.001`. Verify it matches `keras.layers.LayerNormalization` to within a small tolerance.
9. **Capstone B — Fashion MNIST custom loop:** train with a custom loop that displays epoch, iteration, mean training loss and accuracy (updating each iteration), validation loss and accuracy at epoch end, and uses a **different optimizer and learning rate for the upper vs. lower layers**. Wrap the train step in `@tf.function`.
10. Write a one-page "when to go low-level" guide in your own words: which of Phases 3–8 you'd reach for first for each of: a noisy-label regression task, a metric that must be accumulated across batches, a block repeated many times in a paper, a regularizer based on activations, a two-optimizer model, and slow Python-heavy training steps.

### 8.1 Reference: Tracing Gotchas

| What you write | What happens in a TF Function |
|---|---|
| `np.random.rand()` | Evaluated once, at tracing |
| `tf.random.uniform([])` | New value on each call (part of the graph) |
| `print(...)` / counter updates | Only at tracing; use `tf.print` for per-call output |
| `for i in range(n)` | Unrolled at tracing |
| `for i in tf.range(n)` | Becomes a `tf.while_loop` in the graph |
| Python number args | One graph per distinct value |
| Tensor args | One graph per distinct (shape, dtype) |

**Deliverable for Phase 8:** The TF Function rule demonstrations, a timing comparison with/without `@tf.function`, the verified `LayerNormalization` clone, the two-optimizer Fashion MNIST loop, and your one-page decision guide.

---

## Exercises (Must Do)

Adapted from the end-of-section exercises:

| # | Exercise | Skills Practiced |
|---|----------|--------------------|
| 1 | Practice training a deep net on CIFAR10: 20 hidden layers × 100 neurons, He init + ELU, Nadam + early stopping; re-find the learning rate whenever the architecture changes | Default-config training |
| 2 | Add Batch Norm to the CIFAR10 model; compare convergence speed and accuracy. Then try SELU (standardized inputs, LeCun init, dense-only) | Normalization trade-offs |
| 3 | Regularize with alpha dropout; then, without retraining, check whether MC Dropout improves accuracy | MC Dropout |
| 4 | Retrain with 1cycle scheduling; compare training speed and accuracy | LR scheduling |
| 5 | Describe TensorFlow in one sentence, list its main features, and name other popular deep learning libraries | Orientation |
| 6 | Is TensorFlow a drop-in replacement for NumPy? List the main differences | NumPy vs. TF |
| 7 | Do `tf.range(10)` and `tf.constant(np.arange(10))` give the same result? (check dtypes) | dtype awareness |
| 8 | Name six data structures in TensorFlow beyond regular tensors | Data structures |
| 9 | When would you write a custom loss as a function vs. a `Loss` subclass? Same question for a metric (function vs. `Metric` subclass) | Design judgment |
| 10 | When should you create a custom layer vs. a custom model? | Layer vs. Model |
| 11 | What use cases require a custom training loop? Can custom Keras components contain arbitrary Python code, or must they be convertible to TF Functions? List the main TF Function rules | Limits of customization |
| 12 | When would you need a dynamic Keras model, how do you create one, and why not make every model dynamic? | `dynamic=True` / `run_eagerly` |
| 13 | Implement the custom Layer Normalization layer and the two-optimizer Fashion MNIST training loop from Phase 8 | Capstone implementations |

---

## Deliverables Checklist

- [ ] **Phase 1:** Default and self-normalizing configuration tables, three exceptions, seven review answers
- [ ] **Phase 2:** Tensors/variables/NumPy-interop demo + six extra data structures described
- [ ] **Phase 3:** Huber loss (function, closure, subclass) with save/load round trips
- [ ] **Phase 4:** Custom activation/initializer/regularizer/constraint verified + streaming metric (and the precision 40% vs. 50% demo)
- [ ] **Phase 5:** `MyDense`, `MyGaussianNoise`, `ResidualRegressor`, `ReconstructingRegressor` working
- [ ] **Phase 6:** Gradient comparison (hand/finite-difference/tape), `stop_gradient`, custom-gradient `NaN` fix
- [ ] **Phase 7:** Custom training loop (+ validation, constraints, two optimizers)
- [ ] **Phase 8:** TF Function rule demos, timing comparison, LayerNorm clone, two-optimizer Fashion MNIST loop, decision guide
- [ ] **Exercises:** All 13 exercises attempted
- [ ] **Code:** Clean notebook that runs top to bottom without errors

---

## Reference: Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Default DNN Configuration** | He init + ELU, Batch Norm if deep, early stopping (+ℓ2), momentum/RMSProp/Nadam, 1cycle |
| **Self-Normalizing Net** | A dense-only stack using LeCun init + SELU (+ alpha dropout), standardized inputs, no BN |
| **Op / Kernel** | A TensorFlow operation and its device-specific implementation (CPU/GPU/TPU) |
| **`tf.Tensor`** | Immutable multi-dimensional array with shape and dtype |
| **`tf.Variable`** | Mutable tensor (supports `assign()`); used for weights and optimizer state |
| **Sparse / Ragged / String Tensors** | Specialized TF data structures for mostly-zero data, variable-length lists, and byte strings |
| **Huber Loss** | Quadratic for small errors, linear for large ones; robust to outliers |
| **`custom_objects`** | Dictionary mapping saved names to Python objects when loading models with custom components |
| **`get_config()`** | Method returning a component's hyperparameters so Keras can save and recreate it |
| **Streaming (Stateful) Metric** | A metric that accumulates state across batches (e.g., true/false positives) instead of averaging per-batch values |
| **Custom Layer** | A `keras.layers.Layer` subclass with `build()` (create weights) and `call()` (computation) |
| **Custom Model** | A `keras.Model` subclass; a layer plus `compile/fit/evaluate/predict/save` |
| **`add_loss()` / `add_metric()`** | Register losses/metrics computed from model internals (e.g., reconstruction loss) |
| **`tf.GradientTape`** | Context that records operations on variables so gradients can be computed in reverse mode |
| **Jacobian** | Matrix of per-output gradients; `tape.jacobian()` instead of differentiating the sum |
| **`tf.stop_gradient`** | Passes values forward but blocks gradients backward |
| **`@tf.custom_gradient`** | Lets you supply a numerically stable gradient function |
| **Custom Training Loop** | Hand-written epoch/batch loops using a tape and `optimizer.apply_gradients()` |
| **TF Function (`@tf.function`)** | A Python function compiled into an optimized TensorFlow graph |
| **AutoGraph** | Rewrites Python control flow into TF graph ops |
| **Tracing** | Running the function with symbolic tensors to build the graph |
| **Eager vs. Graph Mode** | Eager runs ops immediately; graph mode adds nodes without computing |
| **Dynamic Model** | A model run eagerly (`dynamic=True` / `run_eagerly=True`) instead of being converted to a graph |

---

## Resources

| Resource | Link |
|----------|------|
| TensorFlow Core guides | [tensorflow.org/guide](https://www.tensorflow.org/guide) |
| Keras: custom layers, models, losses & metrics | [keras.io/guides](https://keras.io/guides/) |
| Better performance with `tf.function` | [tensorflow.org/guide/function](https://www.tensorflow.org/guide/function) |
| Introduction to gradients and automatic differentiation | [tensorflow.org/guide/autodiff](https://www.tensorflow.org/guide/autodiff) |
| TensorFlow Model Optimization Toolkit | [tensorflow.org/model_optimization](https://www.tensorflow.org/model_optimization) |
| Papers with Code | [paperswithcode.com](https://paperswithcode.com/) |
| Task 16 reference document | `Task_16_Neural_Networks_with_Keras_and_Training_Deep_Networks.md` |
| Task 14 reference document | `Task_14_Deep_Learning_for_Recommendations.md` |

---

## Suggested Timeline

| Date | Phases |
|------|--------|
| 15/10/2026 | Phase 1–8, excercises |

---
**NOTE:** As in previous tasks, attend daily standups/wrap-ups and keep the tracking sheet updated.