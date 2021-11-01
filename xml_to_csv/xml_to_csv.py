import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
# cnt = 0


def xml_to_csv(path):
    xml_list = []
    # global cnt
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (path + root.find('filename').text,
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text),
                     member[0].text
                     )
            # if member[0].text == 'mask_weared_incorrect':
            #     cnt += 1
            xml_list.append(value)
    column_name = ['img_path', 'x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright', 'classname']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('label.csv', index=None)
    # print(cnt)
    print('Successfully converted xml to csv.')


main()