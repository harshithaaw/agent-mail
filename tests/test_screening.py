from screening import screen_email

# Case 1: obvious spam
result1 = screen_email(
    "FREE VIAGRA CLICK HERE NOW urltoken limited time offer!!!",
    "spam@randomsite.ru"
)
print("Spam case:", result1)

# Case 2: phishing-style
# Phishing-flavored but shouldn't trip the spam classifier's vocabulary

phishing_text = "we noticed a new sign-in from an unrecognized device. click here to secure your account: http://192.168.1.1/secure-login"
result2 = screen_email(phishing_text, "security@totally-not-paypal.ru")
print("Phishing case via screen_email:", result2)
# Case 3: clean
result3 = screen_email(
    "hey, want to grab lunch tomorrow around noon?",
    "friend@gmail.com"
)
print("Clean case:", result3)
sender_email = "security@mybank-alerts.com"

raw_body = """
Urgent notice from your bank.

We noticed unusual activity on your account. Please verify your account 
immediately by clicking the link below to confirm your identity.

http://mybank-alerts.com/verify-now

Thank you,
Account Security Team
"""
print(screen_email(raw_body,sender_email))