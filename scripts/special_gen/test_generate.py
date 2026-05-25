# -*- coding: utf-8 -*-
import sys
import cv2

sys.stdout.reconfigure(encoding='utf-8')

from generate_multi_plate import MultiPlateGenerator

generator = MultiPlateGenerator('plate_model', 'font_model')

# 1. 警用号牌
img = generator.generate_plate_special('湘A9999警', 'white', False)
cv2.imencode('.jpg', img)[1].tofile('test_警用号牌.jpg')
print('OK: 警用号牌')

# 2. 使馆号牌
img = generator.generate_plate_special('使123456', 'black_shi', False)
cv2.imencode('.jpg', img)[1].tofile('test_使馆号牌.jpg')
print('OK: 使馆号牌')

# 3. 领馆号牌
img = generator.generate_plate_special('湘A8888领', 'black', False)
cv2.imencode('.jpg', img)[1].tofile('test_领馆号牌.jpg')
print('OK: 领馆号牌')

# 4. 挂车号牌 (双层黄牌)
img = generator.generate_plate_special('湘A9999挂', 'yellow', True)
cv2.imencode('.jpg', img)[1].tofile('test_挂车号牌.jpg')
print('OK: 挂车号牌')

print()
print('All 4 types generated successfully!')
