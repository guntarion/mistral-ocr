# ocr_pdf.py
# OCR a local PDF using Mistral API and generate human-friendly output
from mistralai import Mistral
import os
import json
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()


def ocr_pdf(pdf_path, save_images=False, output_dir=None):
    """
    Process a PDF file with Mistral OCR and return the results.
    
    Args:
        pdf_path: Path to the PDF file
        save_images: Whether to save images from the OCR results
        output_dir: Directory to save images (if None, a directory will be created)
    
    Returns:
        The OCR response object
    """
    # Get API key from environment
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set.")
    
    print(f"Processing PDF: {pdf_path}")
    client = Mistral(api_key=api_key)
    
    # Upload PDF
    print("Uploading PDF to Mistral API...")
    with open(pdf_path, "rb") as f:
        uploaded_pdf = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr"
        )
    
    # Get signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    # Run OCR
    print("Running OCR processing (this may take a while)...")
    start_time = time.time()
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url}
    )
    
    print(f"OCR processing completed in {time.time() - start_time:.2f} seconds")
    
    # Save images if requested
    if save_images and hasattr(ocr_response, 'pages'):
        if not output_dir:
            # Create default output directory if none provided
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = f"{pdf_name}_images"
        
        save_images_from_ocr(ocr_response, output_dir)
    
    return ocr_response


def save_images_from_ocr(ocr_response, output_dir):
    """Save any images from the OCR response to the specified directory"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_count = 0
    
    # Iterate through pages and save images
    for page in ocr_response.pages:
        if hasattr(page, 'images') and page.images:
            for img in page.images:
                if hasattr(img, 'image_base64') and img.image_base64:
                    image_count += 1
                    img_path = output_path / f"{img.id}"
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(img.image_base64))
                    print(f"Saved image: {img_path}")
    
    if image_count > 0:
        print(f"Saved {image_count} images to {output_dir}")
    else:
        print("No images found in OCR results")


def extract_markdown_from_result(result, include_images=False, images_dir=None):
    """
    Extract markdown content from OCR result with proper formatting
    
    Args:
        result: The OCR response object
        include_images: Whether to include image references in markdown
        images_dir: Directory where images are stored
        
    Returns:
        Formatted markdown string
    """
    pages_content = []
    
    # Handle result object based on its structure
    if hasattr(result, 'pages'):
        # Handle object with pages attribute (like in the example)
        for i, page in enumerate(result.pages):
            page_num = i + 1
            page_content = f"## Page {page_num}\n\n"
            if hasattr(page, 'markdown'):
                page_content += page.markdown
            elif isinstance(page, dict) and "markdown" in page:
                page_content += page["markdown"]
            
            # Handle image references if requested
            if include_images and hasattr(page, 'images') and page.images:
                for img in page.images:
                    if hasattr(img, 'id'):
                        # Replace image references with proper markdown image links
                        img_path = f"{images_dir}/{img.id}" if images_dir else img.id
                        
                        # Look for image references in different formats
                        if f"!{img.id}!" in page_content:
                            page_content = page_content.replace(
                                f"!{img.id}!", f"![Image {img.id}]({img_path})"
                            )
                        
                        if f"![{img.id}]({img.id})" in page_content:
                            page_content = page_content.replace(
                                f"![{img.id}]({img.id})", f"![Image {img.id}]({img_path})"
                            )
                        
                        # Some OCR results might use this format
                        if f"![img-{i}.jpeg](img-{i}.jpeg)" in page_content:
                            page_content = page_content.replace(
                                f"![img-{i}.jpeg](img-{i}.jpeg)", 
                                f"![Image {img.id}]({img_path})"
                            )
            
            pages_content.append(page_content)
    
    elif isinstance(result, dict) and "pages" in result:
        # Handle dictionary with pages key
        for i, page in enumerate(result["pages"]):
            page_num = i + 1
            page_content = f"## Page {page_num}\n\n"
            if "markdown" in page:
                page_content += page["markdown"]
            pages_content.append(page_content)
    
    elif isinstance(result, list):
        # Handle list of pages
        for i, page in enumerate(result):
            page_num = i + 1
            page_content = f"## Page {page_num}\n\n"
            if hasattr(page, "markdown"):
                page_content += page.markdown
            elif isinstance(page, dict) and "markdown" in page:
                page_content += page["markdown"]
            pages_content.append(page_content)
    
    # Join all pages with clear page separators
    return "\n\n" + "\n\n---\n\n".join(pages_content)


def process_and_save_results(pdf_path, save_images=False, output_format="markdown"):
    """
    Process a PDF with OCR and save results in the specified format
    
    Args:
        pdf_path: Path to the PDF file
        save_images: Whether to save images from the OCR results
        output_format: Format to save results (markdown or json)
    """
    # Create output directories
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = f"{pdf_name}_ocr_results"
    images_dir = os.path.join(output_dir, "images") if save_images else None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images directory if needed
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    # Process the PDF
    result = ocr_pdf(pdf_path, save_images=save_images, output_dir=images_dir)
    
    # Save markdown content
    if output_format in ["markdown", "md", "both"]:
        output_md = os.path.join(output_dir, f"{pdf_name}.md")
        markdown_content = extract_markdown_from_result(
            result, 
            include_images=save_images,
            images_dir="./images" if save_images else None
        )
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown content written to {output_md}")
    
    # Save raw JSON response
    if output_format in ["json", "both"]:
        output_json = os.path.join(output_dir, f"{pdf_name}_raw.json")
        # Convert the response to a JSON serializable format
        if hasattr(result, '__dict__'):
            # For custom objects, we need to recursively convert to dict
            def obj_to_dict(obj):
                if hasattr(obj, '__dict__'):
                    return {k: obj_to_dict(v) for k, v in obj.__dict__.items() 
                            if not k.startswith('_')}
                elif isinstance(obj, list):
                    return [obj_to_dict(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: obj_to_dict(v) for k, v in obj.items()}
                else:
                    return obj
            
            result_dict = obj_to_dict(result)
        else:
            result_dict = result
            
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"Raw JSON response written to {output_json}")
    
    return output_dir


def main():
    """Main function to parse arguments and run the OCR process"""
    parser = argparse.ArgumentParser(description="OCR a PDF file using Mistral API")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--save-images", 
        action="store_true", 
        help="Save images from OCR results"
    )
    parser.add_argument(
        "--output-format", 
        choices=["markdown", "json", "both"], 
        default="markdown",
        help="Format to save results (default: markdown)"
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = process_and_save_results(
            args.pdf_path, 
            save_images=args.save_images,
            output_format=args.output_format
        )
        print(f"OCR processing completed successfully. Results saved in {output_dir}")
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":
    main()