import os
import fitz  # PyMuPDF


def extract_text_by_paragraph(pdf_path, min_chunk_size=512):
    """
    Extracts text from a PDF file by paragraphs, combining smaller blocks into cohesive paragraphs,
    removing duplicates, and ensuring minimum chunk size.

    Args:
        pdf_path (str): Path to the PDF file.
        min_chunk_size (int): Minimum size of each text chunk in characters.

    Returns:
        list: List of unique paragraphs extracted from the PDF.
    """
    paragraphs = []
    seen_paragraphs = set()

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text_blocks = page.get_text("blocks")
            current_paragraph = ""

            for block in text_blocks:
                block_text = block[4].strip()
                if block_text:  # Ignore empty blocks
                    if current_paragraph:
                        current_paragraph += " " + block_text
                    else:
                        current_paragraph = block_text

                    # Check if the block ends with a sentence-ending punctuation
                    if block_text.endswith(('.','?','!')):
                        paragraph = current_paragraph.strip()
                        if paragraph not in seen_paragraphs:
                            paragraphs.append((page_number, paragraph))
                            seen_paragraphs.add(paragraph)
                        current_paragraph = ""

            # Append any remaining text as a paragraph
            if current_paragraph:
                paragraph = current_paragraph.strip()
                if paragraph not in seen_paragraphs:
                    paragraphs.append((page_number, paragraph))
                    seen_paragraphs.add(paragraph)

    # Merge paragraphs to meet the minimum chunk size
    merged_paragraphs = []
    current_chunk = ""
    current_pages = []

    for page_number, paragraph in paragraphs:
        if len(current_chunk) < min_chunk_size:
            if current_chunk:
                current_chunk += " " + paragraph
            else:
                current_chunk = paragraph
            # Add page number
            if page_number not in current_pages:
                current_pages.append(page_number)
        else:
            if page_number not in current_pages:
                current_pages.append(page_number)
            page_range = f"{current_pages[0]}-{current_pages[-1]}" if len(current_pages) > 1 else str(current_pages[0])
            if current_chunk not in merged_paragraphs:
                merged_paragraphs.append((page_range, current_chunk.strip()))
            current_chunk = paragraph
            current_pages = [page_number]

    # Append the last chunk
    if current_chunk and current_chunk not in merged_paragraphs:
        page_range = f"{current_pages[0]}-{current_pages[-1]}" if len(current_pages) > 1 else str(current_pages[0])
        merged_paragraphs.append((page_range, current_chunk.strip()))

    return merged_paragraphs


def extract_text_from_each_document(documents_folder, extracted_text_folder):
    # Iterate through each PDF in the documents folder
    for filename in os.listdir(documents_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_folder, filename)

            # Extract paragraphs from the PDF
            paragraphs = extract_text_by_paragraph(pdf_path)

            # Save each paragraph to an individual text file
            for i, (pages, paragraph) in enumerate(paragraphs):
                os.makedirs(f"{extracted_text_folder}/{filename.strip('.pdf')}", exist_ok=True)
                txt_filename = f"{filename.strip('.pdf')}/{os.path.splitext(filename)[0]}_p{i + 1}.txt"
                txt_path = os.path.join(extracted_text_folder, txt_filename)

                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(f"{filename}\n")  # First line: Document name
                    txt_file.write(f"{pages}\n")  # Second line: Page number
                    txt_file.write(f"{paragraph}\n")  # Third line onwards: Paragraph content

    print("Text extraction completed. Paragraphs saved in extracted_text/ folder.")
