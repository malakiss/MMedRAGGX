import argparse
import torch
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from PIL import Image
 
def main(args):
    disable_torch_init()
    
    # Load your fine-tuned DPO model
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.dpo_checkpoint,
        model_base=None,  # Use None if your checkpoint is complete
        model_name="llava-v1.5-13b"
    )
    
    # Read retrieved reports
    print("Reading retrieved reports...")
    with open(args.retrieved_file, 'r') as f:
        data = json.loads(f.readline())
    
    retrieved_reports = data['reference_reports']
    question = data['question']
    image_path = args.img_root + '/' + data['image']
    
    # Construct prompt with retrieved context
    context = "\n\n".join([f"Reference {i+1}: {report}" 
                           for i, report in enumerate(retrieved_reports)])
    
    full_question = f"""Based on the following medical reports:
 
{context}
 
Question: {question}
 
Please provide a detailed answer based on the image and reference reports above."""
    
    # Prepare conversation
    conv = conv_templates["llava_v1"].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + full_question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Load and process image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    # Generate answer
    print("Generating answer...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=1,
            max_new_tokens=args.max_tokens,
            use_cache=True
        )
    
    # Decode output
    outputs = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0].strip()
    
    # Save result
    result = {
        "image": data['image'],
        "question": question,
        "retrieved_reports": retrieved_reports,
        "answer": outputs
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*50)
    print("QUESTION:", question)
    print("="*50)
    print("ANSWER:", outputs)
    print("="*50)
    print(f"\nFull result saved to: {args.output_file}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-checkpoint", type=str, required=True,
                        help="Path to your fine-tuned DPO model checkpoint")
    parser.add_argument("--retrieved-file", type=str, required=True,
                        help="Path to retrieved_output.jsonl from Step 2")
    parser.add_argument("--img-root", type=str, required=True,
                        help="Root directory containing images")
    parser.add_argument("--output-file", type=str, default="vqa_result.json",
                        help="Path to save the final answer")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)