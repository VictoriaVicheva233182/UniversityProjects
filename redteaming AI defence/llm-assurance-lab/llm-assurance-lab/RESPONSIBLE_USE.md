# Responsible use

This project contains attack prompts. They exist to test a system that this
repository itself builds. Please keep it that way.

- Only test systems you own, or systems you have written permission to test.
- Do not point the attack suite at public chatbots or third party APIs.
- All customer data in `data/` is synthetic. Do not add real personal data.
- If you find a weakness in someone else's product while learning from this
  project, report it through their responsible disclosure process.
- The guardrail model is saved with joblib (pickle). Only load model files you
  created yourself.
