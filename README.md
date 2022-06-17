# GUI-Perceptual-Grouping
Detect graphic user interface (GUI) elements from the GUI image and recognize the repetitive layout structure through grouping the elemetns.

## What is it?
On GUIs, we do not just see a collection of separated texts, images, buttons, etc.
Instead,we see perceptual groups of GUI widgets, such as card, list, tab and menu.

<p align="center">
<img src="/data/demo/groups.png" height="300px"/> 
</p>

Although humans can intuitively see perceptual groups of GUI widgets, current computational approaches are rather limited in partitionnig a GUI into meaningful groups of widget elements.
Thus, the GUI Perceptual Grouping is proposed to mimic how human beings perceive the GUI to recognize and gather the atomic GUI elements into structured groups.

### Gestalt Principles
The approch's design uses a well-established psychological theory, the Gestalt principles of perception, for reference to develop its core algorithms.
Gestalt theory systematically explains how humans see the whole rather than individual and unrelated parts.
It includes a set of principles of grouping, among which `connectedness`, `similarity`, `proximity` and `continuity` are the most essential ones.

>Connectedness - We perceive elements connected by uniform visual properties as being more related than those not connected

<p align="center">
<img src="/data/demo/connectedness.png" height="400px"/> 
</p>

>Similarity - Elements are perceptually grouped together if they are similar to each other

<p align="center">
<img src="/data/demo/similarity.png" height="250px"/> 
</p>

>Proximity - When people see an assortment of objects, they tend to perceive objects that are close (proximate) to each other as a group

<p align="center">
<img src="/data/demo/proximity.png" height="300px"/> 
</p>

>Contiuity - Elements arranged in a line or curve are perceived to be more related

<p align="center">
<img src="/data/demo/continuity.png" height="200px"/> 
</p>

### Gestalt in GUI
Gestalt principles also greatly influence UI design, but they may be represented in a disctinct way. 

<p align="center">
<img src="/data/demo/architecture.png" height="300px"/> 
</p>

>Connectedness - Box container that contains multiple widgets within it, and all the enclosed widgets are perceived as in the same group

<p align="center">
<img src="/data/demo/gui-connectedness.png" height="300px"/> 
</p>

>Similarity - Similarity can be observed in aspects of various visual cues, such as size, color, shape or position

<p align="center">
<img src="/data/demo/gui-similarity.png" height="300px"/> 
</p>

>Proximity - Some groups are close to each other and similar in terms of the number and layout of the contained widgets

<p align="center">
<img src="/data/demo/gui-proximity.png" height="300px"/> 
</p>

>Contiuity - Some detection errors are likely to be spotted if a GUI area or a widget aligns with all the widgets in a perceptual group in a line but is not gathered into that group

<p align="center">
<img src="/data/demo/gui-continuity.png" height="300px"/> 
</p>

## How to use it?

The GUI Perceptual Grouping is based on the [UIED](https://github.com/MulongXie/UIED) to detect the GUI element, but both UIED and the grouping approach are unsupervised that requires NO traning process and massive data preparation to deploy extremly easily.

### Dependency
- Python
- Pandas
- sklearn
- OpenCV

> Note that no strict version required for the dependency as long as the program running well in your machine.

### Usage

The approach is purely based on computer vision techniques, it thus only needs the input of a GUI image and then outputs the GUI elements and their grouping.

The basic usage is:
```
gui = GUI(img_file="data/input/2.jpg", output_dir="data/output")
gui.detect_element(is_ocr=True, is_non_text=True, is_merge=True)  
gui.visualize_element_detection() 
gui.recognize_layout() 
gui.visualize_layout_recognition()  
```

## Result

<p align="center">
<img style="width: 80%;" src="/data/demo/result.png"> 
</p>
