from PIL import Image, ImageDraw

COLOR_TABLE = [
    'Red', 'Green', 'Blue', 'Yellow',
    'Purple', 'Orange', 'DarkGreen', 'Purple',
    'YellowGreen', 'Maroon', 'Teal',
    'DarkGoldenrod', 'Peru', 'DarkRed', 'Tan',
    'AliceBlue', 'LightBlue', 'Cyan', 'Teal',
    'SpringGreen', 'SeaGreen', 'Lime', 'DarkGreen',
    'YellowGreen', 'Ivory', 'Olive', 'DarkGoldenrod',
    'Orange', 'Tan', 'Peru', 'Seashell',
    'Coral', 'RosyBrown', 'Maroon', 'DarkRed',
    'WhiteSmoke', 'LightGrey', 'Gray'
] * 10



def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=2):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    # drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill='Black')
    drawObj.text((xmin+3, ymin), text, fill=color)



def show_bbox(img_path, boxes, labels, NAME_TAB, file_name=None, scores=None, 
                matplotlib=False, lb_g=True):
    '''
    img_path: str
    boxes:    list (N, 4)
    labels:   (N)
    NAME_TAB: ['background', 'class_1', 'class_2', ...]
    file_name: 'out.bmp' or None
    scores:   (N)
    '''
    if lb_g: bg_idx = 0
    else: bg_idx = -1

    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # if not isinstance(img, Image.Image):
        # img = transforms.ToPILImage()(img)
    width, height = img.size
    drawObj = ImageDraw.Draw(img)
    for box_id in range(len(boxes)):
        lb = int(labels[box_id])
        if lb > bg_idx:
            box = boxes[box_id]
            # if NAME_TAB is not None:
            if scores is None:
                draw_bbox_text(drawObj, box[1], box[0], box[3], box[2], NAME_TAB[lb], 
                    color=COLOR_TABLE[lb])
            else:
                str_score = str(float(scores[box_id]))[:5]
                str_out = NAME_TAB[lb] + ': ' + str_score
                draw_bbox_text(drawObj, box[1], box[0], box[3], box[2], str_out, 
                    color=COLOR_TABLE[lb])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()
