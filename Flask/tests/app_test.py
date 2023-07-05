import pytest
from app import app as flask_app
import json

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_home(app, client):
    res = client.get('/')
    assert res.status_code == 200
    data = res.data.decode()
    expected = "API Flask OC projet 7"
    assert data == expected

def test_cust_id(app, client):
    res = client.get('/cust_id')
    res_list = json.loads(res.data.decode('utf-8'))
    assert res.status_code == 200
    assert type(res_list) is list
    
def test_predict(app, client):
    customer_list = client.get('/cust_id')
    customer = int(json.loads(customer_list.data.decode('utf-8'))[0])
    res = client.get('/predict', query_string = {'CUSTOMER_ID' : customer})
    res_decode = json.loads(res.data.decode('utf-8'))
    assert res.status_code == 200
    assert type(res_decode['score']) is float