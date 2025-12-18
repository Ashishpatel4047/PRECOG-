# PRECOG-
Classical OCR uses handcrafted features and struggles with noisy backgrounds, font variations, and irregular capitalization. Neural OCR models learn features directly from data using CNNs and sequence models, enabling robust text extraction from CAPTCHA-like images despite distortions and variability.
# Task 0 - The Dataset 
Deep learning needs data. For this task, you will need to synthesize the dataset. Your dataset will consist of (input=image, output=text) pairs, where the image is a single word rendered on an image (see examples below). The Easy Set Text is rendered using a fixed font and capitalization on a plain white background.

      # The Easy Set Text is rendered using a fixed font and capitalization on a plain white background.
           Plain white background
           Fixed font
           Fixed capitalization
           Text = single word per image

     # The Hard Set Multiple fonts, fluctuating capitalization across individual letters, and noisy or textured backgrounds. Ensure diversity in the dataset to test your model’s ability to generalize.
           Multiple fonts
           Random capitalization (HeLLo, wOrLd)
           Noisy / textured backgrounds
     # The Bonus Set This set is used for the bonus Generation task and can be ignored if you intend to skip that task. It borrows all conditions from the hard set with the added condition that if the background          is green, the word is rendered normally, but if the background is red, the word is rendered in reverse. Note: The output does not change. For example, if “hello” is rendered on a red image as “olleh,” your           model should still produce “hello.”
           Green background → word normal
           Red background → word reversed (output still same)
