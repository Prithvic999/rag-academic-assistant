from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


pdf_text = extract_text_from_pdf("data/sample.pdf")

print(pdf_text[:1000])