from GUI import GUI


input_path = 'data/input/7.jpg'
output_root = 'data/output'

gui = GUI(img_file=input_path, output_dir=output_root)
gui.element_detection(True, True, True)
gui.visualize_element_detection()
gui.layout_recognition()
gui.visualize_layout_recognition()
