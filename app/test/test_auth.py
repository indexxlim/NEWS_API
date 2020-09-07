import unittest
import json
from app.test.base import BaseTestCase
from app.main import db
from app.main.model.blacklist import BlacklistToken


def register_user(self):
    return self.client.post(
        '/user',
        data=json.dumps(dict(
            email='example@gmail.com',
            username='username',
            password='123456'
        )),
        content_type='application/json'
    )


def login_user(self):
    return self.client.post(
        '/auth/login',
        data=json.dumps(dict(
            email='example@gmail.com',
            password='123456'
        )),
        content_type='application/json'
    )


class TestAuthBlueprint(BaseTestCase):
    def test_registration(self):
        """ Test for user registration """
        with self.client:
            user_response = register_user(self)
            data = json.loads(user_response.data.decode())
            print(data)
            self.assertTrue(data['status'] == 'success')
            self.assertTrue(data['message'] == 'Successfully registered.')
            self.assertTrue(data['Authorization'])
            self.assertTrue(user_response.content_type == 'application/json')
            self.assertEqual(user_response.status_code, 201)

    def test_registered_with_already_registered_user(self):
        """ Test registration with already registered email"""
        register_user(self)
        with self.client:
            user_response = register_user(self)
            data = json.loads(user_response.data.decode())
            self.assertTrue(data['status'] == 'fail')
            self.assertTrue(
                data['message'] == 'User already exists. Please Log in.')
            self.assertTrue(user_response.content_type == 'application/json')
            self.assertEqual(user_response.status_code, 409)

    def test_registered_user_login(self):
            """ Test for login of registered-user login """
            with self.client:
                # user registration
                user_response = register_user(self)
                response_data = json.loads(user_response.data.decode())
                self.assertTrue(response_data['status'] == 'success')
                self.assertTrue(
                    response_data['message'] == 'Successfully registered.'
                )
                self.assertTrue(response_data['Authorization'])
                self.assertTrue(user_response.content_type == 'application/json')
                self.assertEqual(user_response.status_code, 201)

                # registered user login
                login_response = login_user(self)
                data = json.loads(login_response.data.decode())
                self.assertTrue(data['status'] == 'success')
                self.assertTrue(data['message'] == 'Successfully logged in.')
                self.assertTrue(data['Authorization'])
                self.assertTrue(login_response.content_type == 'application/json')
                self.assertEqual(login_response.status_code, 200)

    def test_non_registered_user_login(self):
        """ Test for login of non-registered user """
        with self.client:
            user_response = login_user(self)
            data = json.loads(user_response.data.decode())
            self.assertTrue(data['status'] == 'fail')
            print(data['message'])
            self.assertTrue(data['message'] == 'email or password does not match.')
            self.assertTrue(user_response.content_type == 'application/json')
            self.assertEqual(user_response.status_code, 401)



    def test_valid_logout(self):
        """ Test for logout before token expires """
        with self.client:
            # user registration
            user_response = register_user(self)
            response_data = json.loads(user_response.data.decode())
            self.assertTrue(response_data['status'] == 'success')
            self.assertTrue(
                response_data['message'] == 'Successfully registered.')
            self.assertTrue(response_data['Authorization'])
            self.assertTrue(user_response.content_type == 'application/json')            
            self.assertEqual(user_response.status_code, 201)

            # registered user login
            login_response = login_user(self)
            login_data = json.loads(login_response.data.decode())
            self.assertTrue(login_data['status'] == 'success')
            self.assertTrue(login_data['message'] == 'Successfully logged in.')
            self.assertTrue(login_data['Authorization'])
            self.assertTrue(login_response.content_type == 'application/json')           
            self.assertEqual(login_response.status_code, 200)

            # valid token logout
            response = self.client.post(
                '/auth/logout',
                headers=dict(
                    Authorization='Bearer ' + json.loads(
                        login_response.data.decode()
                    )['Authorization']
                )
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'success')
            self.assertTrue(data['message'] == 'Successfully logged out.')           
            self.assertEqual(response.status_code, 200)

    def test_valid_blacklisted_token_logout(self):
        """ Test for logout after a valid token gets blacklisted """
        with self.client:
            # user registration
            user_response = register_user(self)
            register_data = json.loads(user_response.data.decode())
            self.assertTrue(register_data['status'] == 'success')
            self.assertTrue(
                register_data['message'] == 'Successfully registered.')
            self.assertTrue(register_data['Authorization'])
            self.assertTrue(user_response.content_type == 'application/json')
            self.assertEqual(user_response.status_code, 201)
            # user login
            login_response = login_user(self)
            login_data = json.loads(login_response.data.decode())
            self.assertTrue(login_data['status'] == 'success')
            self.assertTrue(login_data['message'] == 'Successfully logged in.')
            self.assertTrue(login_data['Authorization'])
            self.assertTrue(login_response.content_type == 'application/json')
            self.assertEqual(login_response.status_code, 200)
            # blacklist a valid token
            blacklist_token = BlacklistToken(
                token=json.loads(login_response.data.decode())['Authorization'])
            db.session.add(blacklist_token)
            db.session.commit()
            # blacklisted valid token logout
            response = self.client.post(
                '/auth/logout',
                headers=dict(
                    Authorization='Bearer ' + json.loads(
                        login_response.data.decode()
                    )['Authorization']
                )
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'fail')
            self.assertTrue(data['message'] == 'Token blacklisted. Please log in again.')
            self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()