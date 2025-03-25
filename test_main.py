import unittest
from main import detect_objects

class TestYOLODetection(unittest.TestCase):
    def test_detect_objects(self):
        # Test mit dem Beispielbild
        image_path = "fu.jpg"
        try:
            objects = detect_objects(image_path)
            self.assertIsInstance(objects, list)  # Überprüfen, ob die Ausgabe eine Liste ist
            if objects:  # Wenn Objekte erkannt wurden
                for obj in objects:
                    self.assertIn("class", obj)
                    self.assertIn("confidence", obj)
                    self.assertIn("box", obj)
        except FileNotFoundError:
            self.fail(f"Test fehlgeschlagen: Bild '{image_path}' nicht gefunden.")

if __name__ == "__main__":
    unittest.main()
