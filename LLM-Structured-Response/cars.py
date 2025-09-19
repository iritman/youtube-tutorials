import os
import instructor
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")
base_url = os.getenv("OPENROUTER_BASE_URL")

client = instructor.from_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url,
    ),
    mode=instructor.Mode.JSON,
)

text = """

هیوندای، i30

دقایقی پیش
2010
180,000 km
اتوماتیک
احراز هویت شده
 بندرعباس
توافقی
لیفان، X60
4
لیفان، X60

دقایقی پیش
1394
238,000 km
دنده ای
 قزوین
650,000,000
تومان
لندرور، رنجرور
4
لندرور، رنجرور

دقایقی پیش
1978
100,000 km
دنده ای
منطقه آزاد
 چالوس
650,000,000
تومان
پژو، 206
3
پژو، 206

دقایقی پیش
1388
216,000 km
تیپ 3
 ری
330,000,000
تومان
اس دبلیو ام، G01
7
اس دبلیو ام، G01

سنجاق شده 
1399
112,000 km
اتوماتیک
 تهران / رسالت
1,370,000,000
تومان
banner
سانگ یانگ، تیوولی
4
سانگ یانگ، تیوولی

سنجاق شده 
2017
79,000 km
آرمور
 تهران / قیطریه
2,280,000,000
تومان
کی ام سی، K7
2
کی ام سی، K7

سنجاق شده 
1402
30,000 km
اتوماتیک
 اسلام شهر
2,150,000,000
تومان


پژو، 206

3 ساعت پیش
1398
60,000 km
تیپ 5
 لامرد
پیش
350,000,000
تومان
ماهانه
3,000,000
تومان
دیگنیتی، پرستیژ
2
دیگنیتی، پرستیژ

4 ساعت پیش
1404
صفر کیلومتر
اتوماتیک
اپال خودرو (شعبه مرکزی)
 تهران / جمهوری
پیش
1,130,000,000
تومان
ماهانه
59,000,000
تومان
پژو، پارس
2
پژو، پارس

4 ساعت پیش
1404
صفر کیلومتر
ELX-XU7P
اپال خودرو (شعبه مرکزی)
 تهران / جمهوری
پیش
327,000,000
تومان
ماهانه
22,000,000
تومان
    """


class Car(BaseModel):
    model: str
    year: int
    mileage: int
    price: int | None = None  # قیمت کل برای خرید نقدی
    down_payment: int | None = None  # مبلغ پیش پرداخت
    monthly_payment: int | None = None  # مبلغ ماهیانه
    is_installment: bool = False
    city: str | None = None
    area: str | None = None

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract information exactly as provided. Do not make up missing information."
            },
            {"role": "user", 
            "content": "Extract: " + text}
        ],
        response_model=List[Car],
    )
    
    import json
    cars_dict = [car.model_dump() for car in response]

    with open('cars.json', 'w', encoding='utf-8') as f:
        json.dump(cars_dict, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(cars_dict)} cars")
except Exception as e:
    print(f"Validation error: {e}")    