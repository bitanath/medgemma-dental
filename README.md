# Medgemma + PaliGemma + MedSigLip Dental Detection

Fine-tune PaliGemma-448 with MedSigLip encoder for detecting and classifying teeth in dental panoramic X-rays.

## 🎯 Project Overview

This project fine-tunes PaliGemma-3B-PT-448 with a MedSigLip-448 vision encoder to detect and classify teeth in dental panoramic X-rays. The model is trained to:

- **Detect** teeth locations using bounding boxes
- **Classify** teeth into 8 fine-grained types or 4 grouped categories
- **Output** Diagnosis with 


## 📚 Finetuned from

- [HuggingFace MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
- [HuggingFace PaliGemma 3B](https://huggingface.co/google/paligemma-3b-pt-448)
- [HuggingFace MedSigLip](https://huggingface.co/google/medsiglip-448)

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{paligemma_dental_detection,
  title={Medgemma + PaliGemma + MedSigLip Dental Detection},
  author={Bitan Nath},
  year={2026},
  url={https://github.com/bitanath/medgemma-dental}
}
```

## 📄 License

This project is for research purposes. Please respect the licenses of:
- PaliGemma (Gemma license)
- MedSigLip (Google Health AI license)
- Any Dental Datasets used to train the model (TBD)

---

**Built for the Medgemma Hackathon c.2026! 🦷✨**
