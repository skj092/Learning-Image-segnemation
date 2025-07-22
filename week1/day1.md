**Module 1: Thresholding**

* **Theoretical Deep Dive:**
    1.  What is the fundamental principle behind intensity-based thresholding for image segmentation?
    2.  Differentiate between global and local (adaptive) thresholding. When would you prefer one over the other?
    3.  How does Otsu's method automatically determine an optimal threshold? Explain the mathematical intuition behind maximizing inter-class variance or minimizing intra-class variance.
    4.  What are the common challenges with thresholding methods (e.g., uneven illumination, multiple objects with similar intensities)?

* **Practical Implementation/Application:**
    1.  **Manual Global Thresholding:**
        * Load a grayscale image (e.g., a simple document scan or an image with clear foreground/background distinction).
        * Write code to apply a manual global threshold. Display the original image, a histogram of its pixel intensities, and the resulting binary (segmented) image.
        * **Experiment:** Try at least 3 different manual threshold values and observe their effects.
    2.  **Otsu's Thresholding:**
        * Apply `cv2.threshold` with `cv2.THRESH_OTSU` to the same image.
        * **Challenge (Conceptual Implementation):** *Without looking at the full OpenCV source,* sketch out the Python pseudo-code or outline the steps you would take to implement Otsu's method from scratch. Focus on the core logic of iterating through possible thresholds and calculating variances. (You don't need to fully code it unless you want an extra challenge, but understanding the steps is key).
    3.  **Adaptive Thresholding:**
        * Find an image with uneven illumination (e.g., a photo taken with a spotlight, or text on a paper with shadows).
        * Apply `cv2.adaptiveThreshold` using both `cv2.ADAPTIVE_THRESH_MEAN_C` and `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`. Experiment with different `blockSize` and `C` values. Display the results.

* **Analytical Challenges:**
    1.  **Comparison:** Compare the results of manual, Otsu's, and adaptive thresholding on your chosen images. For each method, note down:
        * Its effectiveness in separating foreground/background.
        * Its robustness to noise or illumination variations.
        * Scenarios where it performs well and where it fails.
    2.  **Application Scenario:** Propose a real-world scenario where thresholding alone would be sufficient for segmentation and one where it would clearly fail. Explain why.



