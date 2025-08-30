import tkinter as tk
from app import App

def test_app_instantiation():
    try:
        # We can't run mainloop in a headless environment,
        # but we can check if the app object can be created.
        app = App()
        app.destroy()
        print("App instantiated successfully.")
    except tk.TclError as e:
        print(f"Caught expected TclError in headless environment: {e}")
    except Exception as e:
        print(f"Caught unexpected error during app instantiation: {e}")
        raise

if __name__ == "__main__":
    test_app_instantiation()
