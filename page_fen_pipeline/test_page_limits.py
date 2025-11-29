"""
Simple Test Script for Page Limiting Features
Run this to test the new page limiting functionality.
"""

from page_fen_processor import process_pdf_to_page_fens, save_results_to_json


def test_first_3_pages():
    """Test processing only first 3 pages."""
    print("\n" + "="*60)
    print("TEST 1: Processing First 3 Pages")
    print("="*60)
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,  # Don't save crops for testing
        model="gpt-4o",
        max_pages=3  # Only first 3 pages
    )
    
    print(f"\n✅ Success! Processed {len(results)} pages")
    assert len(results) == 3, f"Expected 3 pages, got {len(results)}"
    
    # Check page numbers are correct
    page_nums = [p['page_num'] for p in results]
    assert page_nums == [1, 2, 3], f"Expected pages [1, 2, 3], got {page_nums}"
    print(f"✅ Page numbers correct: {page_nums}")
    
    return results


def test_page_range():
    """Test processing a specific page range."""
    print("\n" + "="*60)
    print("TEST 2: Processing Pages 2-4")
    print("="*60)
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,
        model="gpt-4o",
        start_page=2,
        end_page=4
    )
    
    print(f"\n✅ Success! Processed {len(results)} pages")
    assert len(results) == 3, f"Expected 3 pages, got {len(results)}"
    
    # Check page numbers are correct
    page_nums = [p['page_num'] for p in results]
    assert page_nums == [2, 3, 4], f"Expected pages [2, 3, 4], got {page_nums}"
    print(f"✅ Page numbers correct: {page_nums}")
    
    return results


def test_start_with_limit():
    """Test starting from a page and limiting count."""
    print("\n" + "="*60)
    print("TEST 3: Processing 2 Pages Starting from Page 3")
    print("="*60)
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,
        model="gpt-4o",
        start_page=3,
        max_pages=2
    )
    
    print(f"\n✅ Success! Processed {len(results)} pages")
    assert len(results) == 2, f"Expected 2 pages, got {len(results)}"
    
    # Check page numbers are correct
    page_nums = [p['page_num'] for p in results]
    assert page_nums == [3, 4], f"Expected pages [3, 4], got {page_nums}"
    print(f"✅ Page numbers correct: {page_nums}")
    
    return results


def test_single_page():
    """Test processing a single page."""
    print("\n" + "="*60)
    print("TEST 4: Processing Only Page 5")
    print("="*60)
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,
        model="gpt-4o",
        start_page=5,
        max_pages=1
    )
    
    print(f"\n✅ Success! Processed {len(results)} page(s)")
    assert len(results) == 1, f"Expected 1 page, got {len(results)}"
    
    # Check page number is correct
    assert results[0]['page_num'] == 5, f"Expected page 5, got {results[0]['page_num']}"
    print(f"✅ Page number correct: {results[0]['page_num']}")
    
    return results


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" TESTING PAGE LIMITING FEATURES")
    print("="*70)
    
    try:
        test_first_3_pages()
        test_page_range()
        test_start_with_limit()
        test_single_page()
        
        print("\n" + "="*70)
        print(" ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe page limiting feature is working correctly!")
        print("You can now use max_pages, start_page, and end_page parameters.\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()

