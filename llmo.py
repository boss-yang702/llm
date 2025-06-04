# import jieba
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# from PIL import Image
# import requests
#
#
# model_id = "google/paligemma-3b-pt-224"
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)
#
# image_url = "/content/combined_image.jpg"
# raw_image = Image.open(image_url).convert("RGB")
# prompt = "<image> caption en"
#
#
#
# inputs = processor(prompt, raw_image, return_tensors="pt")
# output = model.generate(**inputs, max_new_tokens=200)
# print(processor.decode(output[0], skip_special_tokens=True))


import os
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="qwen/Qwen-1_8B", trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")