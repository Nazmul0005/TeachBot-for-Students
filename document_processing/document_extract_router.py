import time
import tempfile
import os
from typing import List
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from common.network_responses import NetworkResponse, HTTPCode
from document_processing.document_extract import DocumentProcessor
from document_processing.document_extract_schema import DocumentExtractionResponse

router = APIRouter(prefix="/api/v1", tags=["document-extraction"])
network_response = NetworkResponse()

def save_extracted_text_to_file(filename: str, extracted_text: str) -> str:
    """
    Save extracted text to a .txt file in the data folder
    Returns the path to the saved file
    """
    try:
        # Create data folder if it doesn't exist
        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True)
        
        # Generate output filename
        base_name = Path(filename).stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_{timestamp}_extracted.txt"
        output_path = data_folder / output_filename
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted Text from: {filename}\n")
            f.write(f"# Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Text Length: {len(extracted_text)} characters\n")
            f.write("=" * 80 + "\n\n")
            f.write(extracted_text)
        
        print(f"    Saved extracted text to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"    Error saving extracted text: {str(e)}")
        return ""

@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_text(http_request: Request, files: List[UploadFile] = File(...)):
    """
    Extract text from one or more documents using Document AI.
    Supports various file formats and automatically converts them to PDF.
    Extracted text is automatically saved as .txt files in the data folder.
    """
    start_time = time.time()
    
    try:
        # Validate files
        if not files:
            raise HTTPException(
                status_code=HTTPCode.BAD_REQUEST,
                detail="No files provided"
            )

        # Initialize processor
        processor = DocumentProcessor()
        
        # Use temporary directory instead of creating uploads folder
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = []
            
            # Save uploaded files to temporary directory
            for file in files:
                if not file.filename:
                    raise HTTPException(
                        status_code=HTTPCode.BAD_REQUEST,
                        detail="File must have a filename"
                    )
                
                file_path = os.path.join(temp_dir, file.filename)
                try:
                    # Save file
                    with open(file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)
                    processed_files.append(file_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=HTTPCode.INTERNAL_SERVER_ERROR,
                        detail=f"Error saving file {file.filename}: {str(e)}"
                    )

            # Process all files
            result = processor.process_multiple_files(processed_files)

            # Format response data
            response_data = {
                'total_files': len(processed_files),
                'failed_files': [],
                'successful_files': [],
                'saved_files': []  # New field to track saved text files
            }

            # Organize results into successful and failed files
            for file_result in result['individual_results']:
                if file_result['success']:
                    # Save extracted text to file
                    saved_file_path = save_extracted_text_to_file(
                        file_result['filename'], 
                        file_result['extracted_text']
                    )
                    
                    successful_file_data = {
                        'file_name': file_result['filename'],
                        'text': file_result['extracted_text'],
                        'text_length': len(file_result['extracted_text']),
                        'saved_to': saved_file_path if saved_file_path else None
                    }
                    
                    response_data['successful_files'].append(successful_file_data)
                    
                    # Track saved files
                    if saved_file_path:
                        response_data['saved_files'].append({
                            'original_file': file_result['filename'],
                            'saved_path': saved_file_path,
                            'text_length': len(file_result['extracted_text'])
                        })
                else:
                    response_data['failed_files'].append({
                        'file_name': file_result['filename'],
                        'error': file_result.get('error', 'Unknown error')
                    })

            # Add summary information
            response_data['summary'] = {
                'total_files_processed': len(processed_files),
                'successful_extractions': len(response_data['successful_files']),
                'failed_extractions': len(response_data['failed_files']),
                'files_saved_to_disk': len(response_data['saved_files']),
                'total_characters_extracted': sum(
                    len(f['text']) for f in response_data['successful_files']
                )
            }

            if result['success']:
                message = f"Text extraction completed successfully. {len(response_data['saved_files'])} files saved to data folder."
                return network_response.success_response(
                    http_code=HTTPCode.SUCCESS,
                    message=message,
                    data=response_data,
                    resource=http_request.url.path,
                    start_time=start_time
                )
            else:
                # If all files failed, return error
                if result['successful_files'] == 0:
                    raise HTTPException(
                        status_code=HTTPCode.UNPROCESSABLE_ENTITY,
                        detail="All files failed to process"
                    )
                else:
                    # Some files succeeded, some failed - return partial success
                    message = f"Text extraction completed with some failures. {len(response_data['saved_files'])} files saved to data folder."
                    return network_response.success_response(
                        http_code=HTTPCode.SUCCESS,
                        message=message,
                        data=response_data,
                        resource=http_request.url.path,
                        start_time=start_time
                    )

    except HTTPException:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPCode.INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )

