import json
import boto3
import email
import custom_utilities as utilities


def handler(event, context):
  try:
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    session = boto3.Session()
    s3_session = session.client('s3')
    response = s3_session.get_object(Bucket=bucket, Key=key)

    endpoint_name = 'sms-spam-classifier-mxnet-2021-11-26-00-30-18-850'
    runtime = session.client('runtime.sagemaker')
    vocabulary_length = 9013
    email_obj = email.message_from_bytes(response['Body'].read())
    from_email = email_obj.get('From')
    body = email_obj.get_payload()[0].get_payload()
    print("body: ", body)
    print("from email: ", from_email)
    
    input_mail = [body.strip()]
    print("sanitized email:", input_mail)
    
    hot_encoded_mail = utilities.one_hot_encode(input_mail, vocabulary_length)
    input_mail = utilities.vectorize_sequences(hot_encoded_mail, vocabulary_length)
    print(input_mail)
    
    data = json.dumps(input_mail.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=data)
    print("vectorized email", response)
    
    res = json.loads(response["Body"].read())

    if res['predicted_label'][0][0] == 0:
        label = 'Not Spam'
    else:
        label = 'Spam'
        
    
    score = round(res['predicted_probability'][0][0], 4)
    score = score*100


    message = """
    We received your email sent to '" + str(email_obj.get('To')) + "' with subject '" + str(email_obj.get('Subject')) + "'.\n
    Here is a 240 character sample of the email body:\n\n""" + body[:240] + """\n
    The email was categorized as """ + str(label) + """ with a """ + str(score) + """% confidence."""

    email_client = session.client('ses')
    response_email = email_client.send_email(
        Destination={'ToAddresses': [from_email]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Email Spam Analysis',
            },
        },
        Source=str(email_obj.get('To')),
    )
    print("Response email", response_email)
  except Exception as ex:
       print("Exception: ", ex)
  return {}