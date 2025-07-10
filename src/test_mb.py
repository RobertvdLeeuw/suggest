import musicbrainzngs

import os
from dotenv import load_dotenv
load_dotenv()

# Set up user agent (required by MusicBrainz API)
musicbrainzngs.set_useragent("Suggest: Music Recommender", "1.0", contact=os.environ["EMAIL_ADDR"])
musicbrainzngs.set_rate_limit()
# musicbrainzngs.auth(os.environ["MB_USERNAME"], os.environ["MB_PW"])

def get_song_tags(artist_name, song_title):
    """Get tags/genres for a song by searching artist and title"""
    try:
        # Search for recordings matching the artist and title
        result = musicbrainzngs.search_recordings(
            query=f'artist:"{artist_name}" AND recording:"{song_title}"',
            limit=5
        )
        
        if not result['recording-list']:
            print(f"No recordings found for '{song_title}' by '{artist_name}'")
            return None
        
        # Get the first matching recording
        recording = result['recording-list'][0]
        recording_id = recording['id']
        
        print(f"Found: {recording['title']} by {recording['artist-credit'][0]['artist']['name']}")
        print(f"Recording ID: {recording_id}")
        
        # Get detailed recording info including tags
        detailed_recording = musicbrainzngs.get_recording_by_id(
            recording_id, 
            includes=['tags']
        )
        
        # Extract tags
        recording_data = detailed_recording['recording']
        tags = recording_data.get('tag-list', [])
        
        if tags:
            print(f"\nTags/Genres for '{song_title}':")
            for tag in tags:
                print(f"  - {tag['name']} (count: {tag['count']})")
        else:
            print(f"\nNo tags found for '{song_title}'")
            
        return tags
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_artist_tags(artist_name):
    """Get tags/genres for an artist"""
    try:
        # Search for artist
        result = musicbrainzngs.search_artists(query=artist_name, limit=1)
        
        if not result['artist-list']:
            print(f"No artist found for '{artist_name}'")
            return None
        
        artist = result['artist-list'][0]
        artist_id = artist['id']
        
        # Get detailed artist info with tags
        detailed_artist = musicbrainzngs.get_artist_by_id(
            artist_id, 
            includes=['tags']
        )
        
        artist_data = detailed_artist['artist']
        tags = artist_data.get('tag-list', [])
        
        print(f"\nTags/Genres for artist '{artist_data['name']}':")
        if tags:
            for tag in tags:
                print(f"  - {tag['name']} (count: {tag['count']})")
        else:
            print("No tags found for this artist")
            
        return tags
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example 1: Get tags for a specific song
    print("=== Song Tags Example ===")
    song_tags = get_song_tags("CAN", "Vitamin C")
    
    print("\n" + "="*50)
    
    # Example 2: Get tags for an artist
    print("=== Artist Tags Example ===")
    artist_tags = get_artist_tags("Miles Davis")
    
    print("\n" + "="*50)
    
    # Example 3: Search and display multiple results
    print("=== Multiple Results Example ===")
    try:
        results = musicbrainzngs.search_recordings(
            query='artist:"The Beatles" AND recording:"Yesterday"',
            limit=3
        )
        
        for i, recording in enumerate(results['recording-list']):
            print(f"\nResult {i+1}:")
            print(f"  Title: {recording['title']}")
            print(f"  Artist: {recording['artist-credit'][0]['artist']['name']}")
            print(f"  ID: {recording['id']}")
            
            # Get tags for this recording
            try:
                detailed = musicbrainzngs.get_recording_by_id(
                    recording['id'], 
                    includes=['tags']
                )
                tags = detailed['recording'].get('tag-list', [])
                if tags:
                    print(f"  Tags: {', '.join([tag['name'] for tag in tags[:5]])}")
                else:
                    print("  Tags: None")
            except:
                print("  Tags: Could not retrieve")
                
    except Exception as e:
        print(f"Error in multiple results example: {e}")
