from rembg import remove

input_path = 'C:\Users\admin\Desktop\python-img\input.jpg'
output_path = './output.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)