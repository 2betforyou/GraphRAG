from docx import Document
from pathlib import Path


def main():
    name = input("법령이름: ")
    
    docx_to_text(name)


def docx_to_text(name):
    docx_path = f"./{name}.docx"
    txt_path = f"./{name}.txt" 


    doc = Document(docx_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


    print(text[:5000])

    Path(txt_path).write_text(text, encoding="utf-8")
    print(f"Saved: {txt_path}")



if __name__ == "__main__":
    main()

