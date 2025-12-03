import React,{useState,useRef} from 'react'
import emailjs from '@emailjs/browser';
import Title from '../layouts/Title';
import ContactLeft from './ContactLeft';


const Contact = () =>{ 
  const [username, setUsername] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [errMsg, setErrMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // ===== Configured with EmailJS ====
  const form = useRef();
    const sendEmail = (e) => {
      e.preventDefault();

      // Validate before sending
      if (username === "") {
        setErrMsg("Username is required!");
        return;
      } else if (phoneNumber === "") {
        setErrMsg("Phone number is required!");
        return;
      } else if (email === "") {
        setErrMsg("Please give your Email!");
        return;
      } else if (!emailValidation(email)) {
        setErrMsg("Give a valid Email!");
        return;
      } else if (subject === "") {
        setErrMsg("Plese give your Subject!");
        return;
      } else if (message === "") {
        setErrMsg("Message is required!");
        return;
      }

      console.log("Sending email with:", { username, phoneNumber, email, subject, message });
      console.log("EmailJS Config:", {
        serviceId: process.env.REACT_APP_EMAILJS_SERVICE_ID,
        templateId: process.env.REACT_APP_EMAILJS_TEMPLATE_ID,
        publicKey: process.env.REACT_APP_EMAILJS_PUBLIC_KEY
      });

      // Send email after validation passes
      emailjs
        .sendForm(
          process.env.REACT_APP_EMAILJS_SERVICE_ID || 'service_onnvynq',
          process.env.REACT_APP_EMAILJS_TEMPLATE_ID || 'template_cep73de',
          form.current,
          {
            publicKey: process.env.REACT_APP_EMAILJS_PUBLIC_KEY || 'Y4F-WWiaTvNERBkTg',
          }
        )
        .then(
          () => {
            console.log('SUCCESS!');
            setSuccessMsg(
              `Thank you dear ${username}, Your Messages has been sent Successfully!`
            );
            setErrMsg("");
            setUsername("");
            setPhoneNumber("");
            setEmail("");
            setSubject("");
            setMessage("");
          },
          (error) => {
            console.log('FAILED...', error.text);
            setErrMsg("Failed to send message. Please try again.");
          }
        );

      }
    


  // ========== Email Validation start here ==============
  const emailValidation = () => {
    return String(email)
      .toLowerCase()
      .match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/);
  };
  // ========== Email Validation end here ================

  // const handleSend = (e) => {
  //   e.preventDefault();
    // if (username === "") {
    //   setErrMsg("Username is required!");
    // } else if (phoneNumber === "") {
    //   setErrMsg("Phone number is required!");
    // } else if (email === "") {
    //   setErrMsg("Please give your Email!");
    // } else if (!emailValidation(email)) {
    //   setErrMsg("Give a valid Email!");
    // } else if (subject === "") {
    //   setErrMsg("Plese give your Subject!");
    // } else if (message === "") {
    //   setErrMsg("Message is required!");
    // } else {
    //   setSuccessMsg(
    //     `Thank you dear ${username}, Your Messages has been sent Successfully!`
    //   );
    //   setErrMsg("");
    //   setUsername("");
    //   setPhoneNumber("");
    //   setEmail("");
    //   setSubject("");
    //   setMessage("");
      
    // }
  // };
  return (
    <section
      id="contact"
      className="w-full py-20 border-b-[1px] border-b-black"
    >
      <div className="flex justify-center items-center text-center">
        <Title title="CONTACT" des="Contact With Me" />
      </div>
      <div className="w-full">
        <div className="w-full h-auto flex flex-col lgl:flex-row justify-between">
          <ContactLeft />
          <div className="w-full lgl:w-[60%] h-full py-10 bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] flex flex-col gap-8 p-4 lgl:p-8 rounded-lg shadow-shadowOne">
            <form className="w-full flex flex-col gap-4 lgl:gap-6 py-2 lgl:py-5" ref={form} onSubmit={sendEmail}>
              {errMsg && (
                <p className="py-3 bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] shadow-shadowOne text-center text-orange-500 text-base tracking-wide animate-bounce">
                  {errMsg}
                </p>
              )}
              {/* {successMsg && (
                <p className="py-3 bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] shadow-shadowOne text-center text-green-500 text-base tracking-wide animate-bounce">
                  {successMsg}
                </p>
              )} */}
              <div className="w-full flex flex-col lgl:flex-row gap-10">
                <div className="w-full lgl:w-1/2 flex flex-col gap-4">
                  <p className="text-sm text-gray-400 uppercase tracking-wide">
                    Your name
                  </p>
                  <input
                    onChange={(e) => setUsername(e.target.value)}
                    value={username}
                    className={`${
                      errMsg === "Username is required!" &&
                      "outline-designColor"
                    } contactInput`}
                    type="text"
                    name="user_name"
                  />
                </div>
                <div className="w-full lgl:w-1/2 flex flex-col gap-4">
                  <p className="text-sm text-gray-400 uppercase tracking-wide">
                    Phone Number
                  </p>
                  <input
                    onChange={(e) => setPhoneNumber(e.target.value)}
                    value={phoneNumber}
                    className={`${
                      errMsg === "Phone number is required!" &&
                      "outline-designColor"
                    } contactInput`}
                    type="text"
                    name="user_phone"
                  />
                </div>
              </div>
              <div className="flex flex-col gap-4">
                <p className="text-sm text-gray-400 uppercase tracking-wide">
                  Email
                </p>
                <input
                  onChange={(e) => setEmail(e.target.value)}
                  value={email}
                  className={`${
                    errMsg === "Please give your Email!" &&
                    "outline-designColor"
                  } contactInput`}
                  type="email"
                  name="user_email"
                />
              </div>
              <div className="flex flex-col gap-4">
                <p className="text-sm text-gray-400 uppercase tracking-wide">
                  Subject
                </p>
                <input
                  onChange={(e) => setSubject(e.target.value)}
                  value={subject}
                  className={`${
                    errMsg === "Plese give your Subject!" &&
                    "outline-designColor"
                  } contactInput`}
                  type="text"
                  name="user_subject"
                />
              </div>
              <div className="flex flex-col gap-4">
                <p className="text-sm text-gray-400 uppercase tracking-wide">
                  Message
                </p>
                <textarea
                  onChange={(e) => setMessage(e.target.value)}
                  value={message}
                  className={`${
                    errMsg === "Message is required!" && "outline-designColor"
                  } contactTextArea`}
                  cols="30"
                  rows="8"
                  name="user_message"
                ></textarea>
              </div>
              <div className="w-full">
                <button
                  type="submit"
                  className="w-full h-12 bg-[#141518] rounded-lg text-base text-gray-400 tracking-wider uppercase hover:text-white duration-300 hover:border-[1px] hover:border-designColor border-transparent"
                >
                  Send Message
                </button>
              </div>
              {errMsg && (
                <p className="py-3 bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] shadow-shadowOne text-center text-orange-500 text-base tracking-wide animate-bounce">
                  {errMsg}
                </p>
              )}
              {successMsg && (
                <p className="py-3 bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] shadow-shadowOne text-center text-green-500 text-base tracking-wide animate-bounce">
                  {successMsg}
                </p>
              )}
            </form>
          </div>
        </div>
      </div>
    </section>
  );
   }


export default Contact;


