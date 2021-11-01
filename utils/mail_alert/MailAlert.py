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
from imutils import paths


class MailAlert():
    """
    This is implementation for sending alert mail.
    """

    def __init__(self, **kwargs):
        """
        Constructor.
        """
        self.log = kwargs.get('log', False)
        # receiver email addresses
        # self.alert_path = kwargs.get('mail_receiver')
        files = open(os.path.join('utils', 'mail_receiver.txt'), "r")
        Lines = files.read().split("\n")    
        self.receiver_emails = []
        for line in Lines:
            if line != '':
                self.receiver_emails.append(line)

        # sender email address
        self.sender_email = "aii20facecheckin@gmail.com"
        self.password = "c9aiuser"

        # alert images storage
        self.alert_path = kwargs.get('images')

    def attach_image(self, img_dict):
        """attach file

        Args:
            img_dict (image): attach image

        Returns:
            file: attach image
        """
        # embedded image to mail
        with open(img_dict['path'], 'rb') as file:
            msg_image = MIMEImage(
                file.read(), name=os.path.basename(img_dict['path']))
            msg_image.add_header('Content-ID', '<{}>'.format(img_dict['cid']))
        return msg_image

    def attach_file(self, filename):
        """attach file

        Args:
            filename (image): attach image

        Returns:
            file: attach image
        """
        # encoded html message to attach file
        part = MIMEBase('application', 'octect-stream')
        part.set_payload(open(filename, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename=%s' % os.path.basename(filename))
        return part

    def check_alert(self):
        """check alert image.

        Returns:
            int: number of alert image
        """
        imagePaths = list(paths.list_images(self.alert_path))
        return len(imagePaths) > 0

    def send_alert(self):
        """send alert message by mail
        """
        # send alert mail include 1 image of Unknown
        msg = MIMEMultipart('related')
        msg["Subject"] = Header(
            u'[AII20] [FaceMaskDetection] No Mask Alert', 'utf-8')
        msg["From"] = self.sender_email
        msg["To"] = ','.join(self.receiver_emails)
        msg_alternative = MIMEMultipart('alternative')
        msg_text = MIMEText(
            u'Image not working - maybe next time', 'plain', 'utf-8')
        msg_alternative.attach(msg_text)
        msg.attach(msg_alternative)

        # to add an attachment is just add a MIMEBase object to read a picture locally.
        imagePaths = list(paths.list_images(self.alert_path))
        msg_html = u'<h1>NO MASK ALERT</h1>'
        for (i, img_path) in enumerate(imagePaths):
            image = dict(title=f"image {i}",
                         path=img_path, cid=str(uuid.uuid4()))
            msg_html += '<h3>Image</h3><div dir="ltr">''<img src="cid:{cid}" alt="{alt}"><br></div>'.format(
                alt=cgi.escape(image['title'], quote=True), **image)

            msg.attach(self.attach_image(image))
            msg.attach(self.attach_file(img_path))
            print(img_path)
            os.remove(img_path)
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