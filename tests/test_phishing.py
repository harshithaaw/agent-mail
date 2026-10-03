from scripts.phishing import phishing_score

# Simulated cases
text1 = "your account will be suspended immediately. click here to verify your account urltoken"
urls1 = ["http://totally-not-paypal.ru/login"]
score1, details1 = phishing_score(text1, urls1, "security@paypal.com")
print("Case 1 (should flag):", score1, details1)

text2 = "hey, want to grab lunch tomorrow?"
urls2 = []
score2, details2 = phishing_score(text2, urls2, "friend@gmail.com")
print("Case 2 (should not flag):", score2, details2)

text3 = "your password will expire in 5 days. please update it in your account settings."
urls3 = []
score3, details3 = phishing_score(text3, urls3, "noreply@github.com")
print("Case 3 (legit security email, should be Low):", score3, details3)