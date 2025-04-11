import os
import time
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT")
WHATSAPP_RECIPIENTS = os.getenv("WHATSAPP_RECIPIENTS", "").split(",")

# Alert thresholds
FIRE_CONFIDENCE_THRESHOLD = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "70.0"))
SENSOR_CONFIDENCE_THRESHOLD = float(os.getenv("SENSOR_CONFIDENCE_THRESHOLD", "70.0"))
ADJUSTED_CONFIDENCE_THRESHOLD = float(os.getenv("ADJUSTED_CONFIDENCE_THRESHOLD", "80.0"))

# Cooldown periods (in seconds) to avoid alert flooding
WHATSAPP_COOLDOWN = int(os.getenv("WHATSAPP_COOLDOWN", "300"))  # 5 minutes
SMS_COOLDOWN = int(os.getenv("SMS_COOLDOWN", "600"))  # 10 minutes

# Store last sent timestamps
last_whatsapp_alert = 0
last_sms_alert = 0

# Initialize Twilio client
client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    print("Warning: Twilio credentials not set. Alerts will not be sent.")

def send_whatsapp_alert(message):
    """
    Send WhatsApp alert to configured recipients
    """
    global last_whatsapp_alert
    
    if not client or not TWILIO_WHATSAPP_NUMBER:
        print("Cannot send WhatsApp alert: Twilio not configured")
        return False
    
    current_time = time.time()
    if current_time - last_whatsapp_alert < WHATSAPP_COOLDOWN:
        print(f"WhatsApp alert cooldown in effect. Skipping alert.")
        return False
    
    success = True
    for recipient in WHATSAPP_RECIPIENTS:
        if not recipient:
            continue
            
        try:
            client.messages.create(
                body=message,
                from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
                to=f"whatsapp:{recipient.strip()}"
            )
            print(f"WhatsApp alert sent to {recipient}")
        except Exception as e:
            print(f"Failed to send WhatsApp alert to {recipient}: {e}")
            success = False
    
    if success:
        last_whatsapp_alert = current_time
    
    return success

def send_emergency_sms(message):
    """
    Send emergency SMS alert to the emergency contact
    """
    global last_sms_alert
    
    if not client or not TWILIO_PHONE_NUMBER or not EMERGENCY_CONTACT:
        print("Cannot send SMS alert: Twilio not fully configured")
        return False
    
    current_time = time.time()
    if current_time - last_sms_alert < SMS_COOLDOWN:
        print(f"SMS alert cooldown in effect. Skipping alert.")
        return False
    
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT
        )
        last_sms_alert = current_time
        print(f"Emergency SMS sent to {EMERGENCY_CONTACT}")
        return True
    except Exception as e:
        print(f"Failed to send emergency SMS: {e}")
        return False

def check_thresholds_and_alert(fire_confidence, sensor_confidence, adjusted_confidence, location="Unknown"):
    """
    Check if confidence values exceed thresholds and send appropriate alerts
    """
    alerts_sent = {
        'whatsapp': False,
        'sms': False
    }
    
    # Check for WhatsApp alerts (soft alerts)
    if fire_confidence >= FIRE_CONFIDENCE_THRESHOLD:
        message = f"⚠️ WARNING: Camera detected High Fire confedence ({fire_confidence:.1f}%) at {location}. Please check the system."
        alerts_sent['whatsapp'] = send_whatsapp_alert(message)
    
    if sensor_confidence >= SENSOR_CONFIDENCE_THRESHOLD:
        message = f"⚠️ WARNING: Abnormal sensor readings detected ({sensor_confidence:.1f}%) at {location}. Please check the environment."
        alerts_sent['whatsapp'] = send_whatsapp_alert(message) or alerts_sent['whatsapp']
    
    # Check for emergency SMS alerts
    if adjusted_confidence >= ADJUSTED_CONFIDENCE_THRESHOLD:
        message = f"🔥 EMERGENCY ALERT: Fire detected with high confidence ({adjusted_confidence:.1f}%) at {location}. Immediate action required!"
        alerts_sent['sms'] = send_emergency_sms(message)
    
    return alerts_sent
