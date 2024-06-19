import os


restaurant_dict = {
     'AMBIENCE#GENERAL': 'không gian',
     'DRINKS#PRICES': 'giá tiền đồ uống',
     'DRINKS#QUALITY': 'chất lượng đồ uống',
     'DRINKS#STYLE&OPTIONS': 'lựa chọn đồ uống',
     'FOOD#PRICES': 'giá tiền đồ ăn',
     'FOOD#QUALITY': 'chất lượng đồ ăn',
     'FOOD#STYLE&OPTIONS': 'lựa chọn đồ ăn',
     'LOCATION#GENERAL': 'địa chỉ',
     'RESTAURANT#GENERAL': 'nhà hàng nói chung',
     'RESTAURANT#MISCELLANEOUS': 'vấn đề khác',
     'RESTAURANT#PRICES': 'giá tiền nhà hàng',
     'SERVICE#GENERAL': 'dịch vụ'
}

phone_dict = {
     'BATTERY':'pin',
     'CAMERA':'máy ảnh',
     'DESIGN':'thiết kế',
     'FEATURES':'tính năng',
     'GENERAL':'nói chung',
     'PERFORMANCE':'hiệu suất',
     'PRICE':'giá tiền',
     'SCREEN':'màn hình',
     'SER_ACC':'phục vụ hoặc phụ kiện',
     'STORAGE':'bộ nhớ'
}


beauty_dict = {
    'colour': 'màu sắc',     
    'others': 'vấn đề khác', 
    'packing': 'bao bì',     
    'price': 'giá tiền',     
    'shipping': 'vận chuyển',
    'smell': 'mùi hương',    
    'stayingpower': 'độ bền màu',
    'texture': 'kết cấu'     
}


education_dict = {
    'Behavior': 'hành vi',              # Behavior
    'Curriculum': 'chương trình giảng dạy', # Curriculum
    'Equipment': 'thiết bị',            # Equipment
    'Exercise': 'bài tập',              # Exercise
    'Experience': 'kinh nghiệm',        # Experience
    'General': 'vấn đề chung',                 # General
    'Grading': 'chấm điểm',             # Grading
    'Knowledge': 'kiến thức',           # Knowledge
    'Lecture Material': 'tài liệu giảng dạy', # Lecture Material
    'Suggestion': 'đề xuất',            # Suggestion
    'Teaching Skill': 'kỹ năng giảng dạy' # Teaching Skill
}

technology_dict = {
    'Accessories': 'phụ kiện',       # Accessories
    'Configuration': 'cấu hình',     # Configuration
    'Genuineness': 'tính xác thực',  # Genuineness
    'Model': 'mẫu mã',                  # Model
    'Other': 'vấn đề khác',                 # Other
    'Performance': 'hiệu suất',      # Performance
    'Price': 'giá tiền',               # Price
    'Service': 'phục vụ',            # Service
    'Ship': 'vận chuyển'             # Ship
}


mother_dict = {
    'Genuineness': 'chân thật',  # Genuineness
    'Price': 'giá tiền',               # Price
    'Quality': 'chất lượng',         # Quality
    'Safety': 'an toàn',             # Safety
    'Service': 'phục vụ',            # Service
    'Ship': 'vận chuyển'             # Ship
}

hotel_dict = {
    'FACILITIES#CLEANLINESS': 'vệ sinh cơ sở vật chất',
    'FACILITIES#COMFORT': 'sự thoải mái cơ sở vật chất',
    'FACILITIES#DESIGN&FEATURES': 'thiết kế và tính năng cơ sở vật chất',
    'FACILITIES#GENERAL': 'cơ sở vật chất nói chung',
    'FACILITIES#MISCELLANEOUS': 'vấn đề khác cơ sở vật chất',
    'FACILITIES#PRICES': 'giá tiền cơ sở vật chất',
    'FACILITIES#QUALITY': 'chất lượng cơ sở vật chất',
    'FOOD&DRINKS#MISCELLANEOUS': 'vấn đề khác đồ ăn thức uống',
    'FOOD&DRINKS#PRICES': 'giá tiền đồ ăn thức uống',
    'FOOD&DRINKS#QUALITY': 'chất lượng đồ ăn thức uống',
    'FOOD&DRINKS#STYLE&OPTIONS': 'lựa chọn đồ ăn thức uống',
    'HOTEL#CLEANLINESS': 'vệ sinh khách sạn',
    'HOTEL#COMFORT': 'sự thoải mái khách sạn',
    'HOTEL#DESIGN&FEATURES': 'thiết kế và tính năng khách sạn',
    'HOTEL#GENERAL': 'khách sạn nói chung',
    'HOTEL#MISCELLANEOUS': 'vấn đề khác',
    'HOTEL#PRICES': 'giá tiền khách sạn',
    'HOTEL#QUALITY': 'chất lượng khách hàng',
    'LOCATION#GENERAL': 'địa chỉ',
    'ROOMS#CLEANLINESS': 'vệ sinh phòng',
    'ROOMS#COMFORT': 'sự thoải mái phòng',
    'ROOMS#DESIGN&FEATURES': 'thiết kế và tính năng phòng',
    'ROOMS#GENERAL': 'phòng nói chung',
    'ROOMS#MISCELLANEOUS': 'vấn đề khác của phòng',
    'ROOMS#PRICES': 'giá tiền phòng',
    'ROOMS#QUALITY': 'chất lượng phòng',
    'ROOM_AMENITIES#CLEANLINESS': 'vệ sinh tiện nghi phòng',
    'ROOM_AMENITIES#COMFORT': 'sự thoải mái tiện nghi phòng',
    'ROOM_AMENITIES#DESIGN&FEATURES': 'thiết kế và tính năng tiện nghi phòng',
    'ROOM_AMENITIES#GENERAL': 'tiện nghi phòng nói chung',
    'ROOM_AMENITIES#MISCELLANEOUS': 'vấn đề khác của tiện nghi phòng',
    'ROOM_AMENITIES#PRICES': 'giá tiền tiện nghi phòng',
    'ROOM_AMENITIES#QUALITY': 'chất lượng tiện nghi phòng',
    'SERVICE#GENERAL': 'phục vụ'
}


def mapping_category(domain):
    if domain =="Restaurant":
        return restaurant_dict
    elif domain == "Hotel":
        return hotel_dict
    
    elif domain == "Mother":
        return model_dict
    
    elif domain == "Technology":
        return technology_dict
    
    elif domain == "Education":
        return education_dict
    
    elif domain == "Beauty":
        return beauty_dict
    
    elif domain == "Phone":
        return phone_dict
    
    else:
        print("ERRORRR IN mapping_category FUNCTION: , ", domain)