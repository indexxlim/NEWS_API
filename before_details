set FLASK_APP = connect.py
python -m flask run

WARNING: This is a development server. Do not use it in a production deployment.
set FLASK_ENV = development


--> 사용기
$ python manage.py db init
$ python manage.py db migrate
$ python manage.py db upgrade

정상적으로 생성되었다면 main 폴더 아래 flask_boilerplate_main.db 파일이 생성됩니다.
이후로 database model이 변경될 경우 migrate 과 update 명령을 실행하면 됩니다.

여기까지해서 User 테이블을 만들었습니다.

tree 명령으로 디렉터리를 확인하면 아래와 같이 migrations 폴더가 생성됩니다.

테스트
$ python manage.py test



curl -X GET -H "accept: application/json" "http://192.168.10.9:5000/user"
 
curl -X POST -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"email\": \"ngle01@ngle.co.kr\", \"password\": \"ngle1234\"}"  "http://192.168.10.9:5000/auth/login"

 $ curl -X GET -H "accept: application/json" -H "Authorization:eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NjUzMzMxMzAsImlhdCI6MTU2NTI0NjcyNSwic3ViIjozfQ.YtThtoBeRDrKjzaVi11JGixZFmnBUJeikEgkDbp2x1E" "http://192.168.10.9:5000/user"
 
 curl -X GET -H "accept: application/json" -H 