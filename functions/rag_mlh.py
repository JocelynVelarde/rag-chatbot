from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

def read_doc(directory: str) -> list[str]:
    # Initialize a PyPDFDirectoryLoader object with the given directory
    file_loader = PyPDFDirectoryLoader(directory)

    # Load PDF documents from the directory
    documents = file_loader.load()

    # Extract only the page content from each document
    page_contents = [doc.page_content for doc in documents]

    return page_contents


# Call the function
full_document = read_doc("pdf")