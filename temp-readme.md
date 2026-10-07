 ---
  How to run the Tinker experiment

  0. Add your OpenAI API key to .env

  OPENAI_API_KEY=sk-...

  1. Generate training data (~2 min, costs ~$0.30)

  source venv/bin/activate
  python -m tinker.datagen
  # → writes tinker/data/sycophantic_advisor_train.jsonl (40 examples)

  2. Fine-tune (kicks off the job, polls ~15–30 min, costs ~$1–2)

  python -m tinker.finetune_openai
  # → saves ft model ID to tinker/models.json when done

  # If you close the terminal, check status later with:
  python -m tinker.finetune_openai --status

  3. Run all 3 evaluations + analysis (once fine-tune is done)

  python -m tinker.run_tinker --eval-only
  # Runs: honest, prompted, trained conditions
  # Logs → tinker/logs/   |   Plots → tinker/figures/

  Or run the full pipeline in one shot:
  python -m tinker.run_tinker  # does stages 1-4 end to end

  ---
  What you'll get in tinker/figures/:
  - tinker_coherence_decay.png — 3-line decay chart (the key signal)
  - tinker_coherence_distribution.png — box plot per condition
  - tinker_dimension_profiles.png — bar chart: which dimensions differ (defensiveness is the interesting one)
  - tinker_detection_rate.png — % of sessions CuP flags per condition
  - tinker_summary.txt — numeric summary with coherence gaps

  The core hypothesis to watch: does trained show less defensiveness than prompted (since the trained model can't "see" its objective to strategically deflect questions about it)?
