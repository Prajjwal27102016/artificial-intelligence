import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(title, image):
   """Utility function to display an image."""
   plt.figure(figsize=(8, 8))
   if len(image.shape) == 2:  
       plt.imshow(image, cmap='gray')
   else:  
       plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   plt.title(title)
   plt.axis('off')
   plt.show()


def get_odd_kernel_size(prompt, default=5):
   """Prompt until a valid positive odd integer is given."""
   while True:
       raw = input(f"{prompt} (default: {default}): ").strip()
       if raw == "":
           return default
       if not raw.isdigit():
           print("Please enter a whole number.")
           continue
       value = int(raw)
       if value <= 0:
           print("Kernel size must be positive.")
       elif value % 2 == 0:
           print("Kernel size must be odd.")
       else:
           return value


def get_int(prompt, default):
   """Prompt for an integer, falling back to default on empty/invalid input."""
   raw = input(f"{prompt} (default: {default}): ").strip()
   if raw == "":
       return default
   try:
       return int(raw)
   except ValueError:
       print(f"Invalid number, using default ({default}).")
       return default


def interactive_edge_detection(image_path):
   """Interactive activity for edge detection and filtering."""
   image = cv2.imread(image_path)
   if image is None:
       print("Error: Image not found!")
       return


   # Convert to grayscale
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   display_image("Original Grayscale Image", gray_image)


   print("Select an option:")
   print("1. Sobel Edge Detection")
   print("2. Canny Edge Detection")
   print("3. Laplacian Edge Detection")
   print("4. Gaussian Smoothing")
   print("5. Exit")


   while True:
       choice = input("Enter your choice (1-5): ").strip()


       if choice == "1":

           sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
           sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
           magnitude = cv2.magnitude(sobel_x, sobel_y)          
           combined_sobel = cv2.convertScaleAbs(magnitude)      
           display_image("Sobel Edge Detection", combined_sobel)


       elif choice == "2":

           print("Adjust thresholds for Canny (default: 100 and 200)")
           lower_thresh = get_int("Enter Lower threshold", 100)
           upper_thresh = get_int("Enter Upper threshold", 200)
           edges = cv2.Canny(gray_image, lower_thresh, upper_thresh)
           display_image("Canny Edge Detection", edges)


       elif choice == "3":

           laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
           display_image("Laplacian Edge Detection", np.abs(laplacian).astype(np.uint8))


       elif choice == "4":

           kernel_size = get_odd_kernel_size("Enter kernel size (odd number)")
           blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
           display_image("Gaussian Smoothed Image", blurred)


       elif choice == "5":
           print("Exiting...")
           break


       else:
           print("Invalid choice. Please select a number between 1 and 5.")



interactive_edge_detection('puppy.jpg')
