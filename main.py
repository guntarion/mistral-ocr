# main.py
# OCR a local PDF using Mistral API
from mistralai import Mistral
import os
from dotenv import load_dotenv
load_dotenv()


def ocr_pdf(pdf_path):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set.")
    client = Mistral(api_key=api_key)
    # Upload PDF
    with open(pdf_path, "rb") as f:
        uploaded_pdf = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr"
        )
    # Get signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    # Run OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url}
    )
    return ocr_response


if __name__ == "__main__":
    pdf_path = "e-Procurement System Berbasis Blockchain.pdf"
    result = ocr_pdf(pdf_path)
    output_md = os.path.splitext(pdf_path)[0] + ".md"

    def extract_markdown_from_result(result):
        # If result is a dict with 'pages', extract markdown from each page
        if isinstance(result, dict) and "pages" in result:
            pages = [page.get("markdown", "") for page in result["pages"]]
            return ("\n\n---\n\n").join(pages)
        # If result is a list of page objects (as in your example), extract markdown
        if isinstance(result, list):
            pages = [
                getattr(page, "markdown", "") if hasattr(
                    page, "markdown") else page.get("markdown", "")
                for page in result
            ]
            return ("\n\n---\n\n").join(pages)
        # If result is a string, just return it
        if isinstance(result, str):
            return result
        # Fallback: try to stringify
        return str(result)

    markdown_content = extract_markdown_from_result(result)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"OCR result written to {output_md}")
