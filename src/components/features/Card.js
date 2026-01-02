import React from 'react'
// import { HiArrowRight } from "react-icons/hi";

const Card = ({item:{title,des,icon}}) => {
  return (
    <div className="w-full px-4 sm:px-6 md:px-8 lg:px-10 xl:px-12 h-auto min-h-80 py-6 sm:py-8 md:py-10 rounded-lg shadow-shadowOne flex items-center bg-gradient-to-r from-bodyColor to-[var(--color-card-end)] group hover:bg-gradient-to-b hover:from-black hover:to-[var(--color-panel-start)] transition-colors duration-100 group">
      <div className="h-auto">
        <div className="flex h-full flex-col gap-6 sm:gap-8 md:gap-10">
          <div className="w-10 h-8 flex flex-col justify-between">
        
            {icon ? (
              <span className="text-5xl text-designColor">{icon}</span>
            ) : (
              <>
                <span className="w-full h-[2px] rounded-lg bg-designColor inline-flex"></span>
                <span className="w-full h-[2px] rounded-lg bg-designColor inline-flex"></span>
                <span className="w-full h-[2px] rounded-lg bg-designColor inline-flex"></span>
                <span className="w-full h-[2px] rounded-lg bg-designColor inline-flex"></span>
              </>
            )}
          </div>
          <div className="flex flex-col gap-6">
            <h2 className="text-xl md:text-2xl font-titleFont font-bold text-gray-300">
              {title}
            </h2>
            <p className="base">{des}</p>
            {/* <span className="text-2xl text-designColor">
              <HiArrowRight />
            </span> */}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Card
