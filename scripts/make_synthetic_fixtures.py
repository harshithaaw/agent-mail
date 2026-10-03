"""Generate synthetic sent email fixtures for testing without privacy leaks."""
import json
import os
import random
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime


def generate_synthetic_fixtures():
    rng = random.Random(42)

    categories_data = [
        # Academic (6 emails)
        [
            ("Re: Research assignment update", "Hi Jordan, I have submitted the research assignment for the computer science course. The professor requested a draft of our literature review by Friday. Let me know if you need any additional citations for the thesis section. Thanks, Alex.", "Jordan Smith <jordan@example.com>"),
            ("Re: Lecture notes for university course", "Hi Morgan, Thanks for sharing the university lecture slides for this week. I reviewed the research notes on machine learning algorithms. We should meet after class to discuss our group project. Best, Alex.", "Morgan Lee <morgan@example.org>"),
            ("Re: Assignment guidelines from professor", "Hi Sam, Professor Davis posted the updated assignment criteria on the university portal. We need to complete the literature review before the upcoming lecture. I will compile our findings tonight. Regards, Alex.", "Sam Taylor <sam@example.com>"),
            ("Re: Thesis defense scheduling", "Hi Chris, The university department confirmed our thesis presentation date for next month. Please send over your research section so I can format the slides. Professor Taylor will review the final draft on Monday. Thanks, Alex.", "Chris Evans <chris@example.org>"),
            ("Re: Course syllabus and lecture schedule", "Hi Casey, I checked the university course schedule for the new semester. The professor mentioned that assignment grades will be published weekly. Let's study together after the afternoon lecture. Best, Alex.", "Casey Kim <casey@example.com>"),
            ("Re: University research grant application", "Hi Pat, The university research board approved our project proposal for the next academic year. Professor Miller asked us to submit the detailed assignment breakdown by tomorrow. I will draft the initial summary now. Regards, Alex.", "Pat Drew <pat@example.org>"),
        ],
        # Career (6 emails)
        [
            ("Re: Software Engineer position interview", "Hi Taylor, Thank you for scheduling the software engineer position interview. I am excited to discuss my application and review my updated resume with the hiring team. Please confirm the meeting link for tomorrow. Best, Alex.", "Taylor Reed <taylor@example.com>"),
            ("Re: Job application follow-up", "Hi Jordan, Following up on my job application for the senior developer position. I have attached my latest resume highlighting my experience in system design. I look forward to hearing from the hiring committee soon. Regards, Alex.", "Jordan Smith <jordan@example.com>"),
            ("Re: Interview invitation for lead role", "Hi Morgan, I received the interview invitation for the team lead position. Thank you for considering my application and resume. I am available for a technical discussion on Thursday afternoon. Thanks, Alex.", "Morgan Lee <morgan@example.org>"),
            ("Re: Technical interview preparation", "Hi Sam, Thanks for sharing details about the technical interview stage for this position. The recruiter mentioned that the hiring process includes a system design round. I will review the job requirements again tonight. Best, Alex.", "Sam Taylor <sam@example.com>"),
            ("Re: Position offer and hiring timeline", "Hi Chris, Thank you for the job offer for the data engineer position. I am reviewing the hiring agreement and will return the signed documents shortly. Looking forward to joining the team next month. Regards, Alex.", "Chris Evans <chris@example.org>"),
            ("Re: Recruiter chat regarding job opening", "Hi Casey, It was great speaking with the recruiter about the open developer position. I submitted my resume through the corporate job portal as requested. Please let me know the next steps in the hiring process. Thanks, Alex.", "Casey Kim <casey@example.com>"),
        ],
        # Personal (6 emails)
        [
            ("Re: Weekend hike plans", "Hi Jordan, Sounds like a plan for Saturday morning! I will bring some snacks and water for our trail walk. Let me know if 8 AM works for meeting up at the park entrance. See you soon, Alex.", "Jordan Smith <jordan@example.com>"),
            ("Re: Dinner recipes for Friday", "Hi Morgan, Thanks for recommending that pasta recipe for dinner. I picked up the ingredients from the grocery store yesterday. Can't wait to try cooking it this weekend! Best, Alex.", "Morgan Lee <morgan@example.org>"),
            ("Re: Coffee catchup tomorrow", "Hi Sam, I would love to grab coffee tomorrow afternoon around 3 PM. That new cafe downtown has great hot chocolate. Let me know if that time works for you. Cheers, Alex.", "Sam Taylor <sam@example.com>"),
            ("Re: Book recommendation", "Hi Chris, I finished reading the mystery novel you recommended last week. The ending was completely unexpected and really enjoyable. Do you have any other fiction suggestions for this month? Thanks, Alex.", "Chris Evans <chris@example.org>"),
            ("Re: Birthday gift ideas", "Hi Casey, I was thinking of getting a board game for Jamie's upcoming birthday party. Let me know if you want to split the cost and buy it together. Talk to you later, Alex.", "Casey Kim <casey@example.com>"),
            ("Re: Movie night schedule", "Hi Pat, Friday night works perfectly for watching the new documentary. I can bring some popcorn and fruit to share. Let me know what time everyone is arriving. Best, Alex.", "Pat Drew <pat@example.org>"),
        ],
        # Promotional (6 emails)
        [
            ("Re: Exclusive newsletter discount offer", "Hi Jordan, Thanks for sharing the monthly newsletter featuring the discount offer. I clicked unsubscribe on the marketing list to reduce inbox traffic. Let me know if there are any other promotional sales available. Thanks, Alex.", "Jordan Smith <jordan@example.com>"),
            ("Re: Special offer on office supplies", "Hi Morgan, I saw the promo offer for discounted office supplies in the weekly newsletter. I opted out and selected unsubscribe on that mailing list. We can check their seasonal sale page directly if needed. Best, Alex.", "Morgan Lee <morgan@example.org>"),
            ("Re: Member discount code", "Hi Sam, Thanks for forwarding the special discount code for the store sale. I updated my preferences to unsubscribe from automatic promotional alerts. Let me know if the offer is still active. Regards, Alex.", "Sam Taylor <sam@example.com>"),
            ("Re: Annual sale notification", "Hi Chris, I checked the promotional sale details mentioned in the company newsletter. The 20% discount offer applies to all software tools this week. I will unsubscribe from further advertisement emails now. Thanks, Alex.", "Chris Evans <chris@example.org>"),
            ("Re: Clearance sale and discount deal", "Hi Casey, I noticed the clearance sale announcement and discount offer in the bulletin. I will unsubscribe from these marketing broadcasts after placing our order. Let me know if you want anything else from the sale. Best, Alex.", "Casey Kim <casey@example.com>"),
            ("Re: Weekly newsletter promotion", "Hi Pat, Thank you for sending over the newsletter highlighting the seasonal discount offer. I hit unsubscribe to keep our mailbox clean from marketing spam. Let me know if you see any other good sales. Regards, Alex.", "Pat Drew <pat@example.org>"),
        ],
    ]

    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    emails = []
    idx = 1

    # Thread IDs: first 3 emails share thread_id "1000000000000001" to form a short thread.
    shared_thread_id = "1000000000000001"

    for category_group in categories_data:
        for subject, body, recipient in category_group:
            dt = base_time + timedelta(hours=idx * 3)
            msg_id = f"10000000000000{idx:02d}"
            t_id = shared_thread_id if idx <= 3 else f"20000000000000{idx:02d}"
            rfc_id = f"<synthetic-{idx:02d}@example.com>"

            email_record = {
                "message_id": msg_id,
                "thread_id": t_id,
                "message_id_rfc": rfc_id,
                "sender_name": "Alex Morgan",
                "sender_email": "alex.morgan@example.com",
                "subject": subject,
                "timestamp": dt.isoformat(),
                "timestamp_raw": format_datetime(dt),
                "body": body,
                "recipient": recipient,
                "cc": "",
            }
            emails.append(email_record)
            idx += 1

    output_path = os.path.join("data", "fixtures", "synthetic_sent_emails.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2)

    print(f"Generated {len(emails)} synthetic sent emails -> {output_path}")


if __name__ == "__main__":
    generate_synthetic_fixtures()
