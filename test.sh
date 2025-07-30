
curl -X POST http://localhost:5000/speech/generate-speech \
     -H "Content-Type: application/json" \
     -d '{
           "text": "سلام! صبح شما هم بخیر و پر از طراوت",
           "model": "vits-piper-fa-haaniye_low",
           "speed": 1.0
         }'
