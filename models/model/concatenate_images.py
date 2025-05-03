import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

def concatenate_images(dir1, dir2, text_dir, taxonomy_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of image files from the first directory
    images_dir1 = os.listdir(dir1)

    for image_name in images_dir1:
        # Create full file paths for both directories and the text file
        image_path1 = os.path.join(dir1, image_name)
        image_path2 = os.path.join(dir2, image_name)
        text_file_path = os.path.join(text_dir, os.path.splitext(image_name)[0] + '.txt')
        taxonomy_pred_path = os.path.join(taxonomy_dir, 'pred', os.path.splitext(image_name)[0] + '.txt')
        taxonomy_gt_path = os.path.join(taxonomy_dir, 'gt', os.path.splitext(image_name)[0] + '.txt')

        # Check if the file also exists in the second directory
        if os.path.isfile(image_path1) and os.path.isfile(image_path2):
            # Open both images
            with Image.open(image_path1) as img1, Image.open(image_path2) as img2:
                # Ensure the images have the same height for horizontal concatenation
                if img1.size[1] != img2.size[1]:
                    img2 = img2.resize((int(img2.width * img1.height / img2.height), img1.height))

                # Concatenate the images horizontally
                new_width = img1.width + img2.width
                new_height = img1.height
                concatenated_image = Image.new('RGB', (new_width, new_height))
                concatenated_image.paste(img1, (0, 0))
                concatenated_image.paste(img2, (img1.width, 0))

                # Read text from the corresponding text file if it exists
                text_to_add = ""
                if os.path.isfile(text_file_path):
                    with open(text_file_path, 'r') as text_file:
                        text_to_add = text_file.readline().strip()
                pred_text = ""
                if os.path.isfile(taxonomy_pred_path):
                    with open(taxonomy_pred_path, 'r') as tax_pred_f:
                        pred_text = tax_pred_f.readline().strip()
                gt_text = ""
                if os.path.isfile(taxonomy_gt_path):
                    with open(taxonomy_gt_path, 'r') as tax_gt_f:
                        gt_text = tax_gt_f.readline().strip()


                # Add the text to the top center of the image
                draw = ImageDraw.Draw(concatenated_image)
                font_size = 24  # Adjust the font size as needed
                font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
                font = ImageFont.truetype(font_path, size=font_size)

                # Calculate the text bounding box to determine its width and height
                text_bbox = draw.textbbox((0, 0), text_to_add, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Calculate the position to center the text at the top
                text_x = (new_width - text_width) // 2
                text_y = 10  # Add a small margin from the top

                draw.text((int(new_width/8), int(new_height/3)), f"Pred: {pred_text}", font=font)
                draw.text((int(new_width/8) * 6, int(new_height/3)), f"GT: {gt_text}", font=font)

                # Draw the text onto the image
                draw.text((text_x, text_y), text_to_add, font=font, fill='white')

                # Save the concatenated image to the output directory
                output_path = os.path.join(output_dir, image_name)
                concatenated_image.save(output_path)

                print(f'Saved concatenated image: {output_path}')
        else:
            print(f'Skipping {image_name} as it does not exist in both directories')

def concatenate_text_and_image(image_folder, text_folder, output_folder):

    # Get the list of image files and text files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = sorted(os.listdir(image_folder))
    text_files = sorted(os.listdir(text_folder))

    for img_file, txt_file in zip(image_files, text_files):
        # Load image
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path)

        # Load text
        txt_path = os.path.join(text_folder, txt_file)
        with open(txt_path, 'r') as f:
            caption = f.read().strip()

        # Prepare the font (adjust the font path and size as needed)
        font_size = 24  # Adjust the font size as needed
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=font_size)

        # Create a new image with enough space for both caption and image
        text_height = font_size
        new_image = Image.new('RGB', (image.width, image.height + int(text_height * 1.5)), color=(255, 255, 255))
        new_image.paste(image, (0, int(text_height*1.5)))

        # Draw the caption
        draw = ImageDraw.Draw(new_image)
        draw.text((int(image.width/2), 0), caption, font=font, fill=(0, 0, 0))

        # Save the new image
        output_path = os.path.join(output_folder, img_file)
        new_image.save(output_path)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Concatenate images from two directories with the same names side by side and add text from a text file.')
    parser.add_argument('dir1', type=str, help='Path to the first directory containing images')
    #parser.add_argument('dir2', type=str, help='Path to the second directory containing images')
    parser.add_argument('text_dir', type=str, help='Path to the directory containing text files with captions')
    #parser.add_argument('taxonomy_dir', type=str)
    parser.add_argument('output_dir', type=str, help='Path to the output directory where concatenated images will be saved')

    # Parse the arguments
    args = parser.parse_args()

    # Run the concatenation process
    #concatenate_images(args.dir1, args.dir2, args.text_dir, args.taxonomy_dir, args.output_dir)
    concatenate_text_and_image(args.dir1, args.text_dir, args.output_dir)
if __name__ == "__main__":
    main()
