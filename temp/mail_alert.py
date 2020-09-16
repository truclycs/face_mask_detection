import cgi
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
import os
import smtplib
from email.mime.base import MIMEBase
from email import encoders
import dlib
import cv2
from imutils import paths


class Mail():
    def __init__(self):
        file1 = open('receiver_address.txt', 'r')
        Lines = file1.read().split("\n")
        self.receiver_emails = []
        for line in Lines:
            if line != '':
                self.receiver_emails.append(line)

        self.sender_email = "aii20fmd@gmail.com"
        self.password = "c9aiuser"

    def attach_image(self, img_dict):
        with open(img_dict['path'], 'rb') as file:
            msg_image = MIMEImage(
                file.read(), name=os.path.basename(img_dict['path']))
            msg_image.add_header('Content-ID', '<{}>'.format(img_dict['cid']))
        return msg_image

    def attach_file(self, filename):
        part = MIMEBase('application', 'octect-stream')
        part.set_payload(open(filename, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename=%s' % os.path.basename(filename))
        return part

    def send_alert(self, alert_path):
        msg = MIMEMultipart('related')
        msg["Subject"] = Header(
            u'[AII20] [FaceMaskDetection] Unknown Alert', 'utf-8')
        msg["From"] = self.sender_email
        msg["To"] = ','.join(self.receiver_emails)
        msg_alternative = MIMEMultipart('alternative')
        msg_text = MIMEText(
            u'Image not working - maybe next time', 'plain', 'utf-8')
        msg_alternative.attach(msg_text)
        msg.attach(msg_alternative)

        # text = """\
        #     Hi,
        #     I found unidentified person, please check if you know that guy:

        #     Regards,
        #     Aii20FaceCheckin
        #     """
        # # attache a MIMEText object to save email content
        # message.set_content(text)

        # to add an attachment is just add a MIMEBase object to read a picture locally.
        imagePaths = list(paths.list_images(alert_path))
        msg_html = u'<h1>Some images coming up</h1>'
        for (i, img_path) in enumerate(imagePaths):
            image = dict(title='Image', path=img_path, cid=str(uuid.uuid4()))
            msg_html += '<h3>Image</h3><div dir="ltr">''<img src="cid:{cid}" alt="{alt}"><br></div>'.format(
                alt=cgi.escape(image['title'], quote=True), **image)

            msg.attach(self.attach_image(image))
            msg.attach(self.attach_file(img_path))
        msg_html = MIMEText(msg_html, 'html', 'utf-8')
        msg_alternative.attach(msg_html)

        # Create connection with server and send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(self.sender_email, self.password)
        server.sendmail(self.sender_email,
                        self.receiver_emails, msg.as_string())
        server.quit()


if __name__ == "__main__":
    mymail = mail()
    mymail.send_alert('./alert')
