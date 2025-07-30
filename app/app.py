from flask import Flask

from speech import speech_bp

app = Flask(__name__)
app.register_blueprint(speech_bp, url_prefix="/speech")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
