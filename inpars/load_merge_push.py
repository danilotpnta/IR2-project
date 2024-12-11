from unsloth import FastLanguageModel

def load_and_merge(model_name, checkpoint):
    model: FastLanguageModel = FastLanguageModel.from_pretrained(model_name)
    
    return model