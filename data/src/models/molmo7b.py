import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
from ollama import chat, ChatResponse
from PIL import Image
from transformers import BitsAndBytesConfig


class Molmo:
    def __init__(self, model_name='allenai/Molmo-7B-O-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto'):
        # For 2 x 24 GB. If using 1 x 48 GB or more (lucky you), you can just use device_map="auto"
        device_map = {
            "model.vision_backbone": "cpu",  # Seems to be required to not run out of memory at 48 GB
            "model.transformer.wte": 0,
            "model.transformer.ln_f": 0,
            "model.transformer.ff_out": 1,
        }
        # # For 2 x 24 GB, this works for *only* 38 or 39. Any higher or lower and it'll either only work for 1 token of output or fail completely.
        # switch_point = 38  # layer index to switch to second GPU
        # device_map |= {f"model.transformer.blocks.{i}": 0 for i in range(0, switch_point)}
        # device_map |= {f"model.transformer.blocks.{i}": 1 for i in range(switch_point, 80)}

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        self.model.model.vision_backbone.float()  # vision backbone needs to be in FP32 for this

        self.model = torch.compile(self.model)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        self.prompt = ("Describe the detailed visual characteristics of the person in the photo that could be used to re-identify the "
        "person. Create a scene graph. "
        "Use the JSON format like in the example shown below and only output the JSON: "
        "{"
        "  \"nodes\": ["
        "    { \"id\": \"person\", \"attributes\": [\"...\", \"...\", ...] },"
        "    { \"id\": \"...\", \"attributes\": [\"...\", \"...\", ...] },"
        "    ... "
        "  ],"
        "  \"edges\": ["
        "    { \"source\": \"...\", \"target\": \"...\", \"relation\": \"...\" },"
        "    { \"source\": \"...\", \"target\": \"...\", \"relation\": \"...\" },"
        "    ... "
        "  ]"
        "}"
        "Ensure nodes, attributes, and edges are well-structured. Ensure that the JSON is valid, and do not output "
        "additional information. In the output, use only english language. "
        "nodes consist of an id and an attributes list. edges consist of a source, a target, and a relation."
        "Use only up to 1000 tokens."
)

    @torch.inference_mode()
    def process_image(self, image, max_new_tokens=1024, stop_strings=["<|endoftext|>"]):
        # torch.cuda.empty_cache()

        with Image.open(image) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            inputs = self.processor.process(images=[image], text=self.prompt)
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_new_tokens, stop_strings=stop_strings),
                tokenizer=self.processor.tokenizer
            )
            generated_tokens = output[0, inputs['input_ids'].size(1):]

            graph = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return graph
