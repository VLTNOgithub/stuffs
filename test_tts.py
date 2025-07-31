from gtts import gTTS
import os

tts = gTTS(text="this is a sentence. please turn right in 100 metres... turn left at the next intersection.", lang="en")
tts.save("/tmp/tts_output.mp3")  # Save audio to a temporary file

# Play the audio file
os.system("mpg123 /tmp/tts_output.mp3")