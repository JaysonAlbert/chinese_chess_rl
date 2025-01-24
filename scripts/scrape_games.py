from playwright.sync_api import Playwright, sync_playwright
import json
import time
import re
import os

def extract_moves(page) -> list:
    """Extract moves from the game viewer page"""
    # Wait for the moves to be loaded in the game viewer
    page.wait_for_selector("frame[name='dhtmlxq']")
    game_frame = page.frame("dhtmlxq")
    
    # Wait for moves to be visible
    game_frame.wait_for_selector("#showText")
    
    # Get the moves text
    moves_text = game_frame.locator("#showText").inner_text()
    
    # Parse the moves into a list
    moves = []
    for line in moves_text.split('\n'):
        if '.' in line:  # Line contains a move
            # Remove move number and whitespace
            move = line.split('.')[1].strip()
            if move:
                moves.append(move)
    
    return moves

def scrape_games(playwright: Playwright, num_games: int = 5):
    """Scrape chess games from dpxq.com"""
    # Create directory for games if it doesn't exist
    games_dir = os.path.join("resources", "games")
    os.makedirs(games_dir, exist_ok=True)
    
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    try:
        # Navigate to the search page
        page.goto("http://www.dpxq.com/hldcg/search/")
        
        page.locator("iframe[name=\"search_end_pos\"]").content_frame.get_by_role("link", name="轻舞飞扬").first.click()

        search_frame = page.locator("iframe[name=\"search_end_pos\"]").content_frame

        # Click the initial search link
        time.sleep(2)  # Wait for search results
        
        game_count = 0
        while game_count < num_games:
            # Wait for the table to be visible
            search_frame.locator("#st").wait_for()
            
            # Get all game links from the table
            links = search_frame.locator("#st > tbody > tr > td:nth-child(2) > a")
            count = links.count()
            
            for i in range(1, count):
                if game_count >= num_games:
                    break
                
                try:
                    # Get fresh reference to the link and click it
                    link = links.nth(i)
                    href = link.get_attribute("href")
                    game_id = re.search(r'id=(\d+)', href).group(1)
                    link.click()
                    time.sleep(2)  # Wait for game to load
                    
                    # Switch to game viewer frame and wait for moves text
                    game_frame = page.frame_locator("iframe[name='name_dhtmlxq_search_view']")
                    game_frame.locator("#showText").wait_for()
                    # Extract moves text from textarea
                    moves_text = game_frame.locator("textarea#showText").input_value()
                    if moves_text:
                        # save moves to file
                        game_file = os.path.join(games_dir, f"{game_id}.txt")
                        with open(game_file, "w", encoding='utf-8') as f:
                            f.write(moves_text)
                        game_count += 1
                        print(f"Saved game {game_count}/{num_games} to {game_file}")
                except Exception as e:
                    print(f"Error processing game {i+1}: {e}")
                    continue
            
            if game_count >= num_games:
                break
    
    finally:
        context.close()
        browser.close()

def main():
    with sync_playwright() as playwright:
        scrape_games(playwright, num_games=10)

if __name__ == "__main__":
    main() 