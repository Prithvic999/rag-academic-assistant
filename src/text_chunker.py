from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


pdf_text = extract_text_from_pdf("data/sample.pdf")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200
)

chunks = text_splitter.split_text(pdf_text)


print(f"Total Chunks Created: {len(chunks)}")


print("\n FIRST CHUNK:\n")
for i, chunk in enumerate(chunks):
    print(f"\nCHUNK {i+1}:\n")
    print(chunk)