import React from 'react'
// import { bannerImg } from "../../assets/index";
import { nagu_img } from "../../assets/index";
const RightBanner = () => {
  return (
    <div className="w-full lgl:w-1/2 flex justify-center items-center relative">
      <img
        className="w-[250px] h-[350px] sm:w-[280px] sm:h-[380px] md:w-[350px] md:h-[450px] lgl:w-[500px] lgl:h-[680px] z-10"
        src={nagu_img}
        alt="bannerImg"
      />
      <div className="absolute bottom-0 w-[280px] h-[350px] sm:w-[320px] sm:h-[380px] md:w-[380px] md:h-[450px] lgl:w-[500px] lgl:h-[650px] bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-card-end)] shadow-shadowOne flex justify-center items-center"></div>
    </div>
  );
}

export default RightBanner
