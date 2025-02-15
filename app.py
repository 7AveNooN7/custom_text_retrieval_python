from src.ui import build_ui

def main():
    
    app = build_ui()
    # Uruchamiamy Gradio
    app.launch(debug=True)

if __name__ == "__main__":
    main()
