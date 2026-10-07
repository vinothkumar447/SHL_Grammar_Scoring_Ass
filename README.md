# Grammar Scoring Engine for Spoken English (SHL Hiring Assessment 2026)

Predicts a 1–5 grammar score (MOS Likert rubric) from 45–60 second English speech clips.

**Cross-validation on 732 clean training clips:**
- Random 5-fold: RMSE **0.496**, Pearson **0.872**
- Speaker-grouped 5-fold: RMSE **0.536**, Pearson **0.849**. This is the honest estimate for new speakers.

**Training-set RMSE** (final ensemble, fitted on all training clips): **0.177**

**Expected test RMSE:** about **0.54**. Test speakers are new, and most test clips are the harder short recordings (see below).

## Approach

Each clip is processed by one pretrained speech-recognition model, **NVIDIA Parakeet-TDT-0.6B-v2** (int8 ONNX, run on CPU with `sherpa-onnx`). Two things come out of it:

1. **The transcript, with word timestamps.** From it I compute 52 interpretable features:
   - speech rate and pauses
   - vocabulary richness and word frequency
   - syntax from a spaCy parse (clauses, parse depth, part-of-speech mix, sentence fragments)
   - rule-based grammar-error counts (subject–verb agreement, determiner–noun number, a/an, "did + past tense")
2. **The model's internal representation.** I take the hidden states of encoder layers 4, 8, 12, 16 and 20, average each over time, and get a 5 × 1024 vector per clip.

On top of these, seven simple regressors are trained with 5-fold cross-validation:

| Model | Input |
|---|---|
| Ridge | layer embeddings |
| SVR | layer embeddings (PCA to 256 dimensions) |
| SVR | linguistic features |
| Ridge | character n-gram TF-IDF of the transcript |
| Ridge | embeddings + features |
| Ridge | encoder layer 16 only |
| Ridge | encoder layer 20 only |

Their out-of-fold predictions are combined with a non-negative linear blend. The blend also learns an offset for the ~45 s clips. Final predictions are clipped to the range 1–5.

## Key findings

- **The 37 training clips labelled 0 are a corrupted batch.** The rubric is 1–5, so 0 is outside it. All 37 clips have IDs `audio_5037`–`audio_5073` and use a different WAV header layout from every other file. None of the test files share that layout. One of them (`audio_5044`, label 0) is the same recording as `audio_743`, which is labelled 4.0. These rows are excluded from training.
- **`sample_submission.csv` lists filenames that are mostly not in the test set** (25 of 204). The submission is therefore built from the 216 filenames in `test.csv`.
- **Intermediate encoder layers carry more grammar information than the final layer.** Layer 16 alone beats the final output, and combining five layers works best.
- **The same speakers recur across the training set, but not in the test set.**
  - Speaker embeddings (WeSpeaker ResNet-34) show that 65% of training clips have another training clip with a near-identical voice (cosine > 0.95). Only 1 of the 216 test clips does.
  - So random K-fold flatters the model, because it partly recognises voices.
  - Keeping each speaker group inside one fold gives RMSE 0.536 instead of 0.496.
  - The two single-layer models and the short-clip offset were chosen with speaker-grouped CV. They lower it from 0.542 to 0.536.
- **Most test clips are the shorter ~45 s recordings.** They make up 69% of the test set but only 24% of the training set. The model is less accurate on them (CV RMSE 0.55 vs 0.48 on 60 s clips), so the expected test RMSE is about 0.53–0.54.

### Ablation

Every row below is scored on the same validation folds of the 732 clean clips, so the numbers are directly comparable.

| Step | CV RMSE |
|---|---|
| Transcript-based models only (linguistic features + TF-IDF), trained **with** the label-0 batch | 0.772 |
| Same, label-0 batch **removed** | 0.730 |
| + final-layer speech-model embedding | 0.574 |
| + embeddings from five encoder layers (**final pipeline**) | **0.496** |
| Final pipeline, but trained **with** the label-0 batch | 0.505 |

## Files

| File | What it is |
|---|---|
| `shl_grammar_scoring.ipynb` | Full pipeline: report, EDA, preprocessing, features, models, ablation, evaluation (training RMSE included), plots, submission |
| `submission.csv` | Predictions for the 216 test clips |
| `transcripts_all.jsonl` | Cached ASR output (text and token timestamps) for all 985 clips |
| `emb_all.npz` | Cached encoder embeddings for all 985 clips |
| `spk_emb.npz` | Cached speaker embeddings (used only for the speaker-grouped evaluation) |
| `requirements.txt` | Python dependencies |

## Reproduce

```bash
pip install -r requirements.txt
# put the competition folder (train.csv, test.csv, train/, test/) next to this repo, or set DATA_DIR
jupyter nbconvert --to notebook --execute shl_grammar_scoring.ipynb
```

- The notebook finds `transcripts_all.jsonl`, `emb_all.npz` and `spk_emb.npz` automatically when they sit next to it (or are attached as a Kaggle dataset). It then skips the slow speech step and runs in a few minutes.
- Without them, it downloads the speech model from GitHub (`k2-fsa/sherpa-onnx` releases) and processes every clip. That takes about 1–2 s per clip on a 4-core CPU, and no GPU is needed.

On Kaggle, turn Internet on.

## Limitations

- ASR errors and very disfluent speech add noise.
- Rare extreme scores (1–1.5) are pulled toward the mean.
- The short-clip slice is harder to score.
- Performance drops on unseen speakers.
- The labels contain some noise.

Tried without gain: embeddings from a second speech model (Whisper small.en encoder). Speaker-grouped CV stayed at 0.543, because Whisper overlaps with what Parakeet already captures.

Possible next steps:
- Learn the layer weighting and pooling instead of plain averaging.
- Fine-tune a text transformer on the transcripts.
- Count edits made by a grammar-error-correction model.
